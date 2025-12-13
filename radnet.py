import os
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms


# ==========================
# CONFIG
# ==========================
SEED = 42

BATCH_SIZE = 64          # 4050: prova 64, se OOM -> 32
NUM_EPOCHS = 40
EARLY_STOP_PATIENCE = 7

LR_HEAD = 1e-4           # LR per la testa
LR_LAYER4 = 1e-5         # LR per layer4 (fine-tuning leggero)
WEIGHT_DECAY = 1e-4

VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

NUM_WORKERS = 8          # Fedora+i7: ok 8; se instabile -> 4
USE_AMP = True
FREEZE_ALL_BUT_LAYER4 = True

BEST_MODEL_PATH = "best_radimagenet_resnet50.pth"


# ==========================
# PATH DATASET (robusto)
# ==========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "archive" / "MRI"   # <-- se diverso, cambia qui
DATA_DIR = str(DATA_DIR)


# ==========================
# SEED
# ==========================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # speed
    torch.backends.cudnn.deterministic = False

seed_everything(SEED)


# ==========================
# AMP (API nuova con fallback)
# ==========================
def make_amp():
    """
    Ritorna (GradScaler, autocast, amp_kwargs_builder)
    usando torch.amp quando disponibile, altrimenti torch.cuda.amp.
    """
    try:
        from torch.amp import GradScaler, autocast

        def autocast_ctx(enabled: bool):
            # torch.amp.autocast richiede device_type
            return autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)

        def scaler_ctor(enabled: bool):
            return GradScaler("cuda", enabled=enabled)

        return scaler_ctor, autocast_ctx

    except Exception:
        # fallback vecchio (se torch.amp non c'è)
        from torch.cuda.amp import GradScaler, autocast

        def autocast_ctx(enabled: bool):
            return autocast(enabled=enabled)

        def scaler_ctor(enabled: bool):
            return GradScaler(enabled=enabled)

        return scaler_ctor, autocast_ctx


SCALER_CTOR, AUTOCAST_CTX = make_amp()


# ==========================
# TRANSFORMS (MRI-friendly)
# ==========================
# Nota: evito ColorJitter e flip aggressivi; affine leggero è più sensato su MRI.
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomApply([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
            shear=5
        )
    ], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ==========================
# DATASET: subset con transform indipendente
# ==========================
class SubsetWithTransform(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base_dataset[self.indices[idx]]  # x è PIL.Image (base_dataset transform=None)
        if self.transform is not None:
            x = self.transform(x)  # -> Tensor
        return x, y


# ==========================
# SPLIT STRATIFICATO (senza sklearn)
# ==========================
def stratified_split_indices(targets, val_split=0.15, test_split=0.15, seed=42):
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
        n_train = n_c - n_val - n_test

        train_idx.extend(idx_c[:n_train].tolist())
        val_idx.extend(idx_c[n_train:n_train + n_val].tolist())
        test_idx.extend(idx_c[n_train + n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


# ==========================
# MODEL
# ==========================
def get_model(num_classes: int):
    backbone = torch.hub.load(
        "Warvito/radimagenet-models",
        model="radimagenet_resnet50",
        verbose=True,
        trust_repo=True,
    )

    # Se è una ResNet-style, togliamo la testa per avere feature 2048
    if hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()

    class RadResNet50Classifier(nn.Module):
        def __init__(self, backbone, num_classes):
            super().__init__()
            self.backbone = backbone
            self.pool = nn.AdaptiveAvgPool2d(1)  # fallback se backbone restituisce feature-map
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(2048, num_classes)

        def forward(self, x):
            x = self.backbone(x)
            # supporta sia [B, 2048] che [B, 2048, H, W]
            if x.dim() == 4:
                x = self.pool(x)
                x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    model = RadResNet50Classifier(backbone, num_classes)

    if FREEZE_ALL_BUT_LAYER4:
        for p in model.backbone.parameters():
            p.requires_grad = False
        # sblocca layer4 se esiste
        if hasattr(model.backbone, "layer4"):
            for p in model.backbone.layer4.parameters():
                p.requires_grad = True

    return model


# ==========================
# METRICHE
# ==========================
@torch.no_grad()
def evaluate(model, loader, criterion, device, n_classes):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    per_class_correct = np.zeros(n_classes, dtype=np.int64)
    per_class_total = np.zeros(n_classes, dtype=np.int64)

    for inputs, labels in tqdm(loader, desc="Eval", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(1)

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

    return epoch_loss, epoch_acc, balanced_acc, per_class_correct, per_class_total


# ==========================
# TRAIN (AMP)
# ==========================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp: bool):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with AUTOCAST_CTX(enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


# ==========================
# MAIN
# ==========================
def main():
    assert os.path.isdir(DATA_DIR), f"DATA_DIR non trovato: {DATA_DIR}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset base SENZA transform (PIL -> transform la applico nei subset)
    base_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)
    class_names = base_dataset.classes
    num_classes = len(class_names)
    targets = base_dataset.targets

    print("Classi trovate:", class_names, "| num_classes:", num_classes)

    # Split stratificato
    train_idx, val_idx, test_idx = stratified_split_indices(
        targets, val_split=VAL_SPLIT, test_split=TEST_SPLIT, seed=SEED
    )

    train_dataset = SubsetWithTransform(base_dataset, train_idx, transform=train_transform)
    val_dataset   = SubsetWithTransform(base_dataset, val_idx, transform=val_test_transform)
    test_dataset  = SubsetWithTransform(base_dataset, test_idx, transform=val_test_transform)

    # Conteggi classi sul train
    train_targets = np.array([targets[i] for i in train_idx], dtype=np.int64)
    class_counts = np.bincount(train_targets, minlength=num_classes).astype(np.float32)
    print("Campioni TRAIN per classe:", class_counts)

    # Class weights per loss (inverso frequenza, "soft")
    class_weights = class_counts.sum() / (num_classes * np.clip(class_counts, 1.0, None))
    print("Class weights (train):", class_weights)

    model = get_model(num_classes).to(device)

    # Loss pesata
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    # Sampler per bilanciare i batch (stabilizza molto la loss)
    sample_weights = class_weights[train_targets]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    # DataLoader ottimizzati
    common_loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )

    # prefetch_factor va passato solo se num_workers>0
    def dl_kwargs():
        return {k: v for k, v in common_loader_kwargs.items() if v is not None}

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,   # con sampler non usare shuffle
        **dl_kwargs()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **dl_kwargs()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **dl_kwargs()
    )

    # Optimizer: LR differenziati (head vs layer4)
    param_groups = [{"params": model.fc.parameters(), "lr": LR_HEAD}]

    if hasattr(model.backbone, "layer4"):
        layer4_params = [p for p in model.backbone.layer4.parameters() if p.requires_grad]
        if layer4_params:
            param_groups.append({"params": layer4_params, "lr": LR_LAYER4})

    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    use_amp = USE_AMP and (device.type == "cuda")
    scaler = SCALER_CTOR(enabled=use_amp)

    best_val_balacc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== EPOCH {epoch}/{NUM_EPOCHS} =====")
        print("LRs:", [pg["lr"] for pg in optimizer.param_groups])

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        print(f"Train - loss: {train_loss:.4f} | acc: {train_acc:.4f}")

        val_loss, val_acc, val_balacc, _, _ = evaluate(
            model, val_loader, criterion, device, num_classes
        )
        print(f"Val   - loss: {val_loss:.4f} | acc: {val_acc:.4f} | bal_acc: {val_balacc:.4f}")

        scheduler.step(val_balacc)

        if val_balacc > best_val_balacc:
            best_val_balacc = val_balacc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"--> Nuovo best salvato: {BEST_MODEL_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Nessun miglioramento per {epochs_no_improve} epoche.")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping: balanced accuracy non migliora.")
            break

    # TEST
    print("\nCarico il best model per il test...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    test_loss, test_acc, test_balacc, per_c_corr, per_c_tot = evaluate(
        model, test_loader, criterion, device, num_classes
    )

    print("\n===== TEST =====")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Balanced Accuracy: {test_balacc:.4f}")

    print("\nAccuracy per classe:")
    for i, name in enumerate(class_names):
        if per_c_tot[i] > 0:
            print(f"{name}: {per_c_corr[i]/per_c_tot[i]:.4f} ({per_c_corr[i]}/{per_c_tot[i]})")


if __name__ == "__main__":
    main()