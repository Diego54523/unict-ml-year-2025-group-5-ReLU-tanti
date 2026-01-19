import os
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ==========================
# Fix import "src"
# ==========================
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.radnet import RadNetRunner


# ==========================
# CONFIG
# ==========================
SEED = 42

BATCH_SIZE = 64
NUM_EPOCHS = 50               # spesso basta con 2-stage (meno tempo)
EARLY_STOP_PATIENCE = 10

# --- 2-stage training ---
WARMUP_EPOCHS = 10             # allena SOLO la head
UNFREEZE_LAYER3 = True       # metti True solo se vuoi (piu' rischio instabilita')

# LR (head più alta; backbone molto bassa)
LR_HEAD_WARMUP = 3e-4
LR_HEAD_FINETUNE = 1e-4
LR_LAYER4 = 1e-5
LR_LAYER3 = 5e-6

WEIGHT_DECAY = 1e-4

# Split "fair" con Custom_CNN: 80/20 (val=test)
VAL_SPLIT = 0.20
TEST_SPLIT = 0.0

NUM_WORKERS = 8
USE_AMP = True                # accelera su CUDA (se disponibile)

LABEL_SMOOTHING = 0.0
GRAD_CLIP_NORM = 1.0

# L1 reg (sui parametri allenabili)
L1_LAMBDA = 1e-7              # inizia piccolo; se troppo alto peggiora

# Normalize coerente (train/val/test)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# output features
OUT_DIR = Path(__file__).resolve().parent / "../../features"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_NPZ = OUT_DIR / "radnet_features.npz"
BEST_PATH = OUT_DIR / "best_radnet_for_features.pth"

# dataset
BASE_DIR = Path(__file__).resolve().parent / "../../"
DATA_DIR = str(BASE_DIR / "archive" / "MRI")


# ==========================
# SEED
# ==========================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

seed_everything(SEED)


# ==========================
# DATASET UTILS
# ==========================
class SubsetWithTransform(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base_dataset[self.indices[idx]]  # PIL.Image
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def stratified_split_indices(targets, val_split=0.20, test_split=0.0, seed=42):
    targets = np.asarray(targets)
    classes = np.unique(targets)
    rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []

    for c in classes:
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)

        n_c = len(idx_c)
        n_test = int(round(test_split * n_c))
        n_val = int(round(val_split * n_c))

        if n_test + n_val > n_c:
            n_val = max(n_c - n_test, 0)
            if n_test + n_val > n_c:
                n_test = max(n_c - n_val, 0)

        n_train = n_c - n_val - n_test
        if n_c > 0 and n_train <= 0:
            if n_val > 0:
                n_val -= 1
                n_train += 1
            elif n_test > 0:
                n_test -= 1
                n_train += 1

        train_idx.extend(idx_c[:n_train].tolist())
        val_idx.extend(idx_c[n_train:n_train + n_val].tolist())
        test_idx.extend(idx_c[n_train + n_val:n_train + n_val + n_test].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


# ==========================
# AMP
# ==========================
SCALER_CTOR, AUTOCAST_CTX = RadNetRunner.make_amp()


# ==========================
# STAGE CONTROL (2-stage)
# ==========================
def set_stage_trainable(model: nn.Module, stage: str, unfreeze_layer3: bool):
    """
    stage:
      - "warmup": allena SOLO head (fc)
      - "finetune": allena head + layer4 (+ opzionale layer3)
    """
    # congela tutto
    for p in model.parameters():
        p.requires_grad = False

    # head sempre allenabile
    for p in model.fc.parameters():
        p.requires_grad = True

    if stage == "finetune":
        # sblocca layer4
        if hasattr(model.backbone, "layer4"):
            for p in model.backbone.layer4.parameters():
                p.requires_grad = True

        # opzionale layer3
        if unfreeze_layer3 and hasattr(model.backbone, "layer3"):
            for p in model.backbone.layer3.parameters():
                p.requires_grad = True


def build_optimizer(model: nn.Module, stage: str, unfreeze_layer3: bool):
    """
    Optimizer con param groups coerenti con i requires_grad attuali.
    """
    param_groups = []

    # head
    head_params = [p for p in model.fc.parameters() if p.requires_grad]
    if head_params:
        lr_head = LR_HEAD_WARMUP if stage == "warmup" else LR_HEAD_FINETUNE
        param_groups.append({"params": head_params, "lr": lr_head})

    # layer4
    if hasattr(model.backbone, "layer4"):
        layer4_params = [p for p in model.backbone.layer4.parameters() if p.requires_grad]
        if layer4_params:
            param_groups.append({"params": layer4_params, "lr": LR_LAYER4})

    # layer3
    if unfreeze_layer3 and hasattr(model.backbone, "layer3"):
        layer3_params = [p for p in model.backbone.layer3.parameters() if p.requires_grad]
        if layer3_params:
            param_groups.append({"params": layer3_params, "lr": LR_LAYER3})

    # AdamW
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    return optimizer


def l1_penalty_trainable(model: nn.Module) -> torch.Tensor:
    """
    L1 penalty sui soli parametri allenabili (requires_grad=True).
    """
    l1 = torch.zeros((), device=next(model.parameters()).device)
    for p in model.parameters():
        if p.requires_grad:
            l1 = l1 + p.abs().sum()
    return l1


# ==========================
# TRAIN / EVAL
# ==========================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp: bool):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with AUTOCAST_CTX(enabled=True):
                logits = model(inputs)
                ce = criterion(logits, labels)
                l1 = l1_penalty_trainable(model) if L1_LAMBDA > 0 else 0.0
                loss = ce + (L1_LAMBDA * l1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            ce = criterion(logits, labels)
            l1 = l1_penalty_trainable(model) if L1_LAMBDA > 0 else 0.0
            loss = ce + (L1_LAMBDA * l1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        running_loss += float(loss.item()) * inputs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_classes):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    per_class_correct = np.zeros(n_classes, dtype=np.int64)
    per_class_total = np.zeros(n_classes, dtype=np.int64)

    for inputs, labels in tqdm(loader, desc="Eval", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, labels)

        running_loss += float(loss.item()) * inputs.size(0)
        preds = logits.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for c in range(n_classes):
            mask = (labels == c)
            per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
            per_class_total[c] += mask.sum().item()

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)

    recalls = []
    for c in range(n_classes):
        if per_class_total[c] > 0:
            recalls.append(per_class_correct[c] / per_class_total[c])
    balanced_acc = float(np.mean(recalls)) if len(recalls) else 0.0

    return epoch_loss, epoch_acc, balanced_acc


# ==========================
# FEATURE EXTRACTION
# ==========================
@torch.no_grad()
def extract_features(model, loader, device):
    """
    Estrae feature 2048-dim PRIMA del classificatore (fc):
    backbone -> pool+flatten (se 4D)
    """
    model.eval()
    feats_list, labels_list = [], []

    for inputs, labels in tqdm(loader, desc="Extract", leave=False):
        inputs = inputs.to(device, non_blocking=True)

        x = model.backbone(inputs)
        if x.dim() == 4:
            x = model.pool(x)
            x = torch.flatten(x, 1)

        feats_list.append(x.detach().cpu().numpy())
        labels_list.append(labels.numpy())

    feats = np.concatenate(feats_list, axis=0) if feats_list else np.empty((0, 2048), dtype=np.float32)
    labs = np.concatenate(labels_list, axis=0) if labels_list else np.empty((0,), dtype=np.int64)
    return feats, labs


def main():
    assert os.path.isdir(DATA_DIR), f"DATA_DIR non trovato: {DATA_DIR}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    base_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)
    class_names = base_dataset.classes
    targets = base_dataset.targets
    num_classes = len(class_names)

    print("Classi trovate:", class_names, "| num_classes:", num_classes)

    # split stratificato
    train_rel, val_rel, test_rel = stratified_split_indices(
        targets, val_split=VAL_SPLIT, test_split=TEST_SPLIT, seed=SEED
    )
    if TEST_SPLIT == 0.0:
        test_rel = list(val_rel)

    # transforms coerenti
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = SubsetWithTransform(base_dataset, train_rel, transform=train_transform)
    val_dataset = SubsetWithTransform(base_dataset, val_rel, transform=val_test_transform)
    test_dataset = SubsetWithTransform(base_dataset, test_rel, transform=val_test_transform)

    # class weights SOLO nella loss (niente sampler)
    train_targets = np.array([targets[i] for i in train_rel], dtype=np.int64)
    class_counts = np.bincount(train_targets, minlength=num_classes).astype(np.float32)
    print("Campioni TRAIN per classe:", class_counts.astype(int).tolist())

    class_weights = class_counts.sum() / (num_classes * np.clip(class_counts, 1.0, None))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    # loader kwargs
    common_loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    def dl_kwargs():
        return {k: v for k, v in common_loader_kwargs.items() if v is not None}

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **dl_kwargs())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, **dl_kwargs())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **dl_kwargs())

    # modello
    runner = RadNetRunner(
        data_dir=DATA_DIR,
        seed=SEED,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        num_workers=NUM_WORKERS,
        use_amp=USE_AMP,
    )
    model = runner.get_model(num_classes).to(device)

    # AMP
    use_amp = USE_AMP and (device.type == "cuda")
    scaler = SCALER_CTOR(enabled=use_amp)

    # ==========================
    # 2-STAGE TRAINING
    # ==========================
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Scheduler su VAL LOSS (stabile)
    # (lo creiamo dopo l'optimizer; verra' ricreato quando cambia stage)
    scheduler = None
    optimizer = None

    for epoch in range(1, NUM_EPOCHS + 1):
        # stage selection
        stage = "warmup" if epoch <= WARMUP_EPOCHS else "finetune"

        # quando entri in finetune (o al primo giro), imposta trainable + optimizer
        if epoch == 1:
            set_stage_trainable(model, stage="warmup", unfreeze_layer3=UNFREEZE_LAYER3)
            optimizer = build_optimizer(model, stage="warmup", unfreeze_layer3=UNFREEZE_LAYER3)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2
            )
            print(f"\n[Stage] warmup: alleno SOLO head per {WARMUP_EPOCHS} epoche")

        if epoch == WARMUP_EPOCHS + 1:
            set_stage_trainable(model, stage="finetune", unfreeze_layer3=UNFREEZE_LAYER3)
            optimizer = build_optimizer(model, stage="finetune", unfreeze_layer3=UNFREEZE_LAYER3)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2
            )
            print("\n[Stage] finetune: alleno head + layer4" + (" + layer3" if UNFREEZE_LAYER3 else ""))

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(f"\n===== EPOCH {epoch}/{NUM_EPOCHS} | stage={stage} =====")
        print("LRs:", lrs, "| L1_LAMBDA:", L1_LAMBDA)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        val_loss, val_acc, val_balacc = evaluate(
            model, val_loader, criterion, device, num_classes
        )

        print(f"Train - loss: {train_loss:.4f} | acc: {train_acc:.4f}")
        print(f"Val   - loss: {val_loss:.4f} | acc: {val_acc:.4f} | bal_acc: {val_balacc:.4f}")

        # scheduler su val_loss
        scheduler.step(val_loss)

        # early stopping su val_loss (stabile)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(BEST_PATH))
            print(f"--> Nuovo best salvato (val_loss): {BEST_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Nessun miglioramento per {epochs_no_improve} epoche.")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping: val loss non migliora.")
            break

    # load best
    print("\nCarico il best model per feature extraction...")
    model.load_state_dict(torch.load(str(BEST_PATH), map_location=device))

    # estrazione feature (deterministica)
    train_feat_ds = SubsetWithTransform(base_dataset, train_rel, transform=val_test_transform)
    val_feat_ds = SubsetWithTransform(base_dataset, val_rel, transform=val_test_transform)
    test_feat_ds = SubsetWithTransform(base_dataset, test_rel, transform=val_test_transform)

    train_feat_loader = DataLoader(train_feat_ds, batch_size=BATCH_SIZE, shuffle=False, **dl_kwargs())
    val_feat_loader = DataLoader(val_feat_ds, batch_size=BATCH_SIZE, shuffle=False, **dl_kwargs())
    test_feat_loader = DataLoader(test_feat_ds, batch_size=BATCH_SIZE, shuffle=False, **dl_kwargs())

    train_feats, train_labels = extract_features(model, train_feat_loader, device)
    val_feats, val_labels = extract_features(model, val_feat_loader, device)
    test_feats, test_labels = extract_features(model, test_feat_loader, device)

    print("\nShapes:")
    print("train_feats:", train_feats.shape, "train_labels:", train_labels.shape)
    print("val_feats  :", val_feats.shape, "val_labels  :", val_labels.shape)
    print("test_feats :", test_feats.shape, "test_labels :", test_labels.shape)
    if TEST_SPLIT == 0.0:
        print("Nota: TEST_SPLIT=0.0 -> test set uguale al validation set.")

    np.savez_compressed(
        OUT_NPZ,
        train_feats=train_feats,
        train_labels=train_labels,
        val_feats=val_feats,
        val_labels=val_labels,
        test_feats=test_feats,
        test_labels=test_labels,
        class_names=np.array(class_names, dtype=object),
        mean=np.array(IMAGENET_MEAN, dtype=np.float32),
        std=np.array(IMAGENET_STD, dtype=np.float32),
        l1_lambda=np.array([L1_LAMBDA], dtype=np.float32),
        warmup_epochs=np.array([WARMUP_EPOCHS], dtype=np.int64),
        unfreeze_layer3=np.array([int(UNFREEZE_LAYER3)], dtype=np.int64),
    )
    print(f"\nFeature salvate in: {OUT_NPZ}")


if __name__ == "__main__":
    main()