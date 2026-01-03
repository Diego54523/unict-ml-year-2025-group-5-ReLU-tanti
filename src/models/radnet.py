# radnet.py
# Feature extraction + train/val/test for RadImageNet ResNet50 with:
# - pretty output
# - AMP new API (no warnings) with fallback
# - imbalanced dataset handling via WeightedRandomSampler (train only)
# - ordinal class remapping + ordinal metrics (MAE, QWK)
# - optional gridSearch on reduced dataset with OOM-proof cleanup and smaller batch size

import os
import gc
import time
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms

from MLP_Softmax_Class import MLP_Softmax_Classifier

# =========================================================
# MODEL 
# =========================================================
class RadResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module | None = None, dropout_p: float = 0.5):
        super().__init__()
        self.backbone = backbone if backbone is not None else self.get_backbone()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = MLP_Softmax_Classifier(
            in_features=2048,
            hidden_units=512,      # puoi cambiare (256 / 512 / 1024)
            out_classes=num_classes
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if x.dim() == 4:
            x = self.pool(x)
            x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def get_backbone() -> nn.Module:
        backbone = torch.hub.load(
            "Warvito/radimagenet-models",
            model="radimagenet_resnet50",
            verbose=True,
            trust_repo=True,
        )
        if hasattr(backbone, "fc"):
            backbone.fc = nn.Identity()  # type: ignore
        return backbone  # type: ignore


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministico (come nel tuo nuovo codice)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# AMP 
# =========================================================
def make_amp():
    """
    Supporta:
      - torch.amp.autocast / torch.amp.GradScaler (API nuova)
      - fallback torch.cuda.amp (API vecchia)
    """
    try:
        from torch.amp import autocast as amp_autocast # type: ignore
        from torch.amp import GradScaler as AmpGradScaler # type: ignore

        def autocast_ctx(device: str, enabled: bool): # type: ignore
            return amp_autocast(device_type=device, dtype=torch.float16, enabled=enabled)

        def scaler_ctor(device: str, enabled: bool): # type: ignore
            return AmpGradScaler(device, enabled=enabled)

        return autocast_ctx, scaler_ctor

    except Exception:
        from torch.cuda.amp import autocast as cuda_autocast
        from torch.cuda.amp import GradScaler as CudaGradScaler

        def autocast_ctx(device: str, enabled: bool):
            return cuda_autocast(enabled=enabled)

        def scaler_ctor(device: str, enabled: bool):
            return CudaGradScaler(enabled=enabled)

        return autocast_ctx, scaler_ctor


AUTOCAST_CTX, SCALER_CTOR = make_amp()


# =========================================================
# PATH / CONFIG 
# =========================================================
BEST_MODEL_PATH = "best_radimagenet_resnet50.pth"


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "../../archive" / "MRI"
DATA_DIR = str(DATA_DIR)

NUM_CLASSES = 4

BATCH_SIZE = 64
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

WEIGHT_DECAY = 1e-4
DROPOUT_P = 0.5
LEARNING_RATE = 1e-3

# split/loader (servono per pipeline)
SEED = 42
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
NUM_WORKERS = 8


# =========================================================
# Classi ordinali
# =========================================================
ORDINAL_CLASS_ORDER = [
    "NonDemented",
    "ModerateDemented",
    "MildDemented",
    "VeryMildDemented",
]


# =========================================================
# TRANSFORMS (MRI-friendly)
# =========================================================
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


# =========================================================
# DATASET HELPERS
# =========================================================
class SubsetWithTransform(Dataset):
    def __init__(self, base_dataset, indices, transform=None, remap_targets: Optional[Dict[int, int]] = None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform
        self.remap_targets = remap_targets  # mapping old_idx -> ordinal_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base_dataset[self.indices[idx]]  # PIL.Image, class idx (ImageFolder)
        if self.remap_targets is not None:
            y = self.remap_targets[int(y)]
        if self.transform is not None:
            x = self.transform(x)
        return x, int(y)


def stratified_split_indices(
    targets, val_split: float = 0.15, test_split: float = 0.15, seed: int = 42
):
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


# =========================================================
# Feature extraction: congelo backbone, alleno solo FC
# =========================================================
def freeze_for_feature_extraction(model: RadResNet50Classifier) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


# =========================================================
# Metriche ordinali
# =========================================================
def mae_ordinal(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(preds.astype(np.float32) - targets.astype(np.float32))))


def quadratic_weighted_kappa(preds: np.ndarray, targets: np.ndarray, n_classes: int) -> float:
    """
    QWK (no sklearn).
    """
    preds = preds.astype(int)
    targets = targets.astype(int)

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(targets, preds):
        O[t, p] += 1.0

    act_hist = np.bincount(targets, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(preds, minlength=n_classes).astype(np.float64)

    E = np.outer(act_hist, pred_hist)
    E = E / np.sum(E) if np.sum(E) > 0 else E

    O = O / np.sum(O) if np.sum(O) > 0 else O

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    denom = float((n_classes - 1) ** 2) if n_classes > 1 else 1.0
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / denom

    num = np.sum(W * O)
    den = np.sum(W * E) if np.sum(W * E) > 0 else 1.0
    return float(1.0 - num / den)


# =========================================================
# EVAL (con autocast su CUDA per ridurre VRAM)
# =========================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, n_classes: int) -> Dict[str, Any]:
    model.eval()
    use_amp = (device.type == "cuda")

    running_loss = 0.0
    correct = 0
    total = 0

    per_class_correct = np.zeros(n_classes, dtype=np.int64)
    per_class_total = np.zeros(n_classes, dtype=np.int64)

    all_preds = []
    all_tgts = []

    for inputs, labels in tqdm(loader, desc="val", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with AUTOCAST_CTX("cuda", enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += float(loss.item()) * inputs.size(0)
        preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.append(preds.detach().cpu().numpy())
        all_tgts.append(labels.detach().cpu().numpy())

        for c in range(n_classes):
            mask = (labels == c)
            per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
            per_class_total[c] += mask.sum().item()

    loss_epoch = running_loss / max(total, 1)
    acc = correct / max(total, 1)

    recalls = []
    for c in range(n_classes):
        if per_class_total[c] > 0:
            recalls.append(per_class_correct[c] / per_class_total[c])
    bal_acc = float(np.mean(recalls)) if len(recalls) else 0.0

    preds_np = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    tgts_np = np.concatenate(all_tgts) if all_tgts else np.array([], dtype=np.int64)

    ord_mae = mae_ordinal(preds_np, tgts_np) if len(preds_np) else 0.0
    qwk = quadratic_weighted_kappa(preds_np, tgts_np, n_classes) if len(preds_np) else 0.0

    return dict(
        loss=loss_epoch,
        acc=acc,
        bal_acc=bal_acc,
        ord_mae=ord_mae,
        qwk=qwk,
        per_class_correct=per_class_correct,
        per_class_total=per_class_total
    )


# =========================================================
# PRETTY PRINT
# =========================================================
def fmt_pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"


def print_epoch_line(epoch: int, epochs: int, lr: float, tr: Dict[str, float], va: Dict[str, Any], dt: float, best: bool):
    flag = " ★BEST" if best else ""
    print(
        f"[{epoch:03d}/{epochs:03d}] "
        f"lr={lr:.2e} | "
        f"train loss {tr['loss']:.4f} acc {fmt_pct(tr['acc'])} | "
        f"val loss {va['loss']:.4f} acc {fmt_pct(va['acc'])} "
        f"bal {fmt_pct(va['bal_acc'])} | "
        f"ordMAE {va['ord_mae']:.3f} QWK {va['qwk']:.3f} | "
        f"{dt:5.1f}s{flag}"
    )


# =========================================================
# TRAIN ONE EPOCH
# =========================================================
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp: bool,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="train", leave=False)
    steps = 0

    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with AUTOCAST_CTX("cuda", enabled=True):
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

        bs = inputs.size(0)
        running_loss += float(loss.item()) * bs
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += bs

        steps += 1
        if total > 0:
            pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{correct/total:.4f}")

        if max_steps is not None and steps >= max_steps:
            break

    loss_epoch = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return dict(loss=loss_epoch, acc=acc)


# =========================================================
# FIT con early stopping (monitor su QWK -> migliore per ordinal)
# =========================================================
def fit(
    model: RadResNet50Classifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_classes: int,
    epochs: int,
    early_patience: int,
    lr: float,
    weight_decay: float,
    max_steps_per_epoch: Optional[int] = None,
) -> Tuple[RadResNet50Classifier, Dict[str, Any]]:
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_amp = (device.type == "cuda")
    scaler = SCALER_CTOR("cuda", enabled=use_amp) if device.type == "cuda" else None

    best_state = None
    best_score = -1e9
    best_val = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            max_steps=max_steps_per_epoch,
        )

        va = evaluate(model, val_loader, criterion, device, n_classes)
        dt = time.time() - t0

        score = float(va["qwk"])
        if not np.isfinite(score):
            score = float(va["bal_acc"])

        improved = score > best_score
        if improved:
            best_score = score
            best_val = copy.deepcopy(va)
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, BEST_MODEL_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print_epoch_line(epoch, epochs, lr, tr, va, dt, improved)

        if epochs_no_improve >= early_patience:
            print(f"Early stopping: nessun miglioramento per {early_patience} epoche.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # cleanup optimizer/scaler references (aiuta in grid search)
    del optimizer
    if scaler is not None:
        del scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return model, {"best_score": best_score, "best_val": best_val}


# =========================================================
# GRID SEARCH (dataset ridotto + epoche “corte”, OOM-proof)
# =========================================================
@dataclass
class GridResult:
    params: Dict[str, float]
    best_score: float


def run_grid_search(
    base_dataset,
    train_idx: List[int],
    val_idx: List[int],
    remap_targets: Dict[int, int],
    n_classes: int,
    device: torch.device,
) -> GridResult:
    rng = np.random.default_rng(SEED)

    def reduce_indices(indices: List[int], frac: float) -> List[int]:
        indices = np.array(indices, dtype=np.int64) # type: ignore
        keep = max(1, int(round(frac * len(indices))))
        if keep >= len(indices):
            return indices.tolist() # type: ignore
        chosen = rng.choice(indices, size=keep, replace=False)
        return chosen.tolist()

    # riduzione aggressiva
    train_small_idx = reduce_indices(train_idx, frac=0.10)
    val_small_idx = reduce_indices(val_idx, frac=0.30)

    train_ds = SubsetWithTransform(base_dataset, train_small_idx, transform=train_transform, remap_targets=remap_targets)
    val_ds = SubsetWithTransform(base_dataset, val_small_idx, transform=val_test_transform, remap_targets=remap_targets)

    # sampler bilanciato sul subset
    train_targets = np.array([remap_targets[int(base_dataset.targets[i])] for i in train_small_idx], dtype=np.int64)
    class_counts = np.bincount(train_targets, minlength=n_classes).astype(np.float32)
    class_weights = class_counts.sum() / (n_classes * np.clip(class_counts, 1.0, None))
    sample_weights = class_weights[train_targets]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double), # type: ignore
        num_samples=len(sample_weights),
        replacement=True,
    )

    # IMPORTANT: batch più piccolo SOLO per gridSearch per evitare OOM
    GRID_BATCH_SIZE = 16  # metti 8 se hai ancora OOM

    train_loader = DataLoader(
        train_ds, batch_size=GRID_BATCH_SIZE, sampler=sampler, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"), persistent_workers=(NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=GRID_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"), persistent_workers=(NUM_WORKERS > 0)
    )

    # griglia piccola attorno ai tuoi valori
    lr_grid = [LEARNING_RATE * 0.3, LEARNING_RATE, LEARNING_RATE * 3.0]
    wd_grid = [WEIGHT_DECAY * 0.3, WEIGHT_DECAY, WEIGHT_DECAY * 3.0]
    dp_grid = [max(0.0, DROPOUT_P - 0.2), DROPOUT_P, min(0.9, DROPOUT_P + 0.2)]

    GRID_EPOCHS = 6
    MAX_STEPS_PER_EPOCH = 20

    best = GridResult(params={}, best_score=-1e9)
    combos = [(lr, wd, dp) for lr in lr_grid for wd in wd_grid for dp in dp_grid]

    print(f"\n[GridSearch] {len(combos)} combinazioni (subset + cap steps/epoch={MAX_STEPS_PER_EPOCH}, bs={GRID_BATCH_SIZE})")

    for lr, wd, dp in combos:
        model = None
        try:
            print(f"\n[GridSearch] lr={lr:.2e} wd={wd:.2e} dropout={dp:.2f}")

            model = RadResNet50Classifier(num_classes=n_classes, dropout_p=dp).to(device)
            freeze_for_feature_extraction(model)

            _, hist = fit(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                n_classes=n_classes,
                epochs=GRID_EPOCHS,
                early_patience=max(2, EARLY_STOPPING_PATIENCE // 3),
                lr=lr,
                weight_decay=wd,
                max_steps_per_epoch=MAX_STEPS_PER_EPOCH,
            )

            score = float(hist["best_score"])
            if score > best.best_score:
                best = GridResult(params={"lr": lr, "weight_decay": wd, "dropout_p": dp}, best_score=score)

        finally:
            # cleanup aggressivo per VRAM tra combo
            if model is not None:
                del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    print(f"\n[GridSearch] BEST params={best.params} | best_score={best.best_score:.4f}\n")
    return best


# =========================================================
# MAIN
# =========================================================
def main(gridSearch: bool = False):
    assert os.path.isdir(DATA_DIR), f"DATA_DIR non trovato: {DATA_DIR}"
    seed_everything(SEED)

    # Suggerimento anti-frammentazione (opzionale):
    # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)
    found_classes = base_dataset.classes
    found_targets = np.array(base_dataset.targets, dtype=np.int64)

    print(f"Classi trovate: {found_classes} | num_classes: {len(found_classes)}")
    assert len(found_classes) == NUM_CLASSES, f"NUM_CLASSES={NUM_CLASSES} ma dataset ha {len(found_classes)} classi."

    # Controllo classi attese
    missing = [c for c in ORDINAL_CLASS_ORDER if c not in found_classes]
    if missing:
        raise ValueError(f"Nel dataset mancano classi attese: {missing}. Classi trovate: {found_classes}")

    # remap idx ImageFolder -> idx ordinale
    class_to_idx = base_dataset.class_to_idx  # name -> idx attuale
    ordinal_name_to_idx = {name: i for i, name in enumerate(ORDINAL_CLASS_ORDER)}
    remap_targets = {class_to_idx[name]: ordinal_name_to_idx[name] for name in ORDINAL_CLASS_ORDER}

    # targets ordinali
    targets_ordinal = np.array([remap_targets[int(t)] for t in found_targets], dtype=np.int64)

    # split stratificato sugli ordinal targets
    train_idx, val_idx, test_idx = stratified_split_indices(
        targets_ordinal, val_split=VAL_SPLIT, test_split=TEST_SPLIT, seed=SEED
    )

    # dataset
    train_ds = SubsetWithTransform(base_dataset, train_idx, transform=train_transform, remap_targets=remap_targets)
    val_ds = SubsetWithTransform(base_dataset, val_idx, transform=val_test_transform, remap_targets=remap_targets)
    test_ds = SubsetWithTransform(base_dataset, test_idx, transform=val_test_transform, remap_targets=remap_targets)

    # sampler bilanciato sul train full
    train_targets = np.array([targets_ordinal[i] for i in train_idx], dtype=np.int64)
    class_counts = np.bincount(train_targets, minlength=NUM_CLASSES).astype(np.float32)

    print("\nDistribuzione TRAIN (ordine ordinale):")
    for i, name in enumerate(ORDINAL_CLASS_ORDER):
        print(f"  {i}: {name:>16s} -> {int(class_counts[i])} campioni")

    class_weights = class_counts.sum() / (NUM_CLASSES * np.clip(class_counts, 1.0, None))
    sample_weights = class_weights[train_targets]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double), # type: ignore
        num_samples=len(sample_weights),
        replacement=True
    )

    common_loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False, **common_loader_kwargs) # type: ignore
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **common_loader_kwargs) # type: ignore
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **common_loader_kwargs) # type: ignore

    # iperparametri (default)
    lr = LEARNING_RATE
    wd = WEIGHT_DECAY
    dp = DROPOUT_P

    # grid search opzionale
    if gridSearch:
        best = run_grid_search(
            base_dataset=base_dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            remap_targets=remap_targets,
            n_classes=NUM_CLASSES,
            device=device
        )
        lr = float(best.params["lr"])
        wd = float(best.params["weight_decay"])
        dp = float(best.params["dropout_p"])

    # model (feature extraction)
    model = RadResNet50Classifier(num_classes=NUM_CLASSES, dropout_p=dp).to(device)
    freeze_for_feature_extraction(model)

    print("\n=== TRAIN (feature extraction: backbone frozen, only head) ===")
    model, hist = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_classes=NUM_CLASSES,
        epochs=EPOCHS,
        early_patience=EARLY_STOPPING_PATIENCE,
        lr=lr,
        weight_decay=wd,
        max_steps_per_epoch=None,
    )

    print("\nBest checkpoint salvato in:", BEST_MODEL_PATH)
    if hist["best_val"] is not None:
        bv = hist["best_val"]
        print(
            f"Best VAL -> loss {bv['loss']:.4f} | acc {fmt_pct(bv['acc'])} | "
            f"bal {fmt_pct(bv['bal_acc'])} | ordMAE {bv['ord_mae']:.3f} | QWK {bv['qwk']:.3f}"
        )

    # TEST
    print("\n=== TEST (best model) ===")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    criterion = nn.NLLLoss()
    te = evaluate(model, test_loader, criterion, device, NUM_CLASSES)

    print(
        f"TEST -> loss {te['loss']:.4f} | acc {fmt_pct(te['acc'])} | "
        f"bal {fmt_pct(te['bal_acc'])} | ordMAE {te['ord_mae']:.3f} | QWK {te['qwk']:.3f}"
    )

    print("\nAccuracy per classe (ordine ordinale):")
    for i, name in enumerate(ORDINAL_CLASS_ORDER):
        tot = int(te["per_class_total"][i])
        cor = int(te["per_class_correct"][i])
        if tot > 0:
            print(f"  {i}: {name:>16s} -> {cor / tot:6.3f} ({cor}/{tot})")
        else:
            print(f"  {i}: {name:>16s} -> N/A (0 campioni)")

    # cleanup finale
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # True = grid search veloce (subset + cap steps/epoch + bs ridotto)
    main(gridSearch=False)
