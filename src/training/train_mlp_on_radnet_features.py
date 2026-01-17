# src/training/train_mlp_on_radnet_features.py

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.MLP_Softmax_Class import MLP_Softmax_Classifier
from src.data.Average_for_Batch_Value import AverageValueMeter


# ==========================
# PATHS
# ==========================
SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_NPZ = (ROOT / "features" / "radnet_features.npz").resolve()
OUT_WEIGHTS = (SCRIPT_DIR / "weights_Softmax_MLP" / "mlp_on_radnet.pth").resolve()

# ==========================
# CONFIG
# ==========================
SEED = 42
TEST_SPLIT = 0.20  # usato solo se NPZ contiene 'features'/'labels'

BATCH_SIZE = 256
EPOCHS = 50
LR = 0.01
MOMENTUM = 0.9

USE_CLASS_WEIGHTS = True


# ==========================
# Utils
# ==========================
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FeaturesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def stratified_train_test_split(X: np.ndarray, y: np.ndarray, test_split: float, seed: int, num_classes: int):
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y))

    train_idx = []
    test_idx = []

    for c in range(num_classes):
        idx_c = idx_all[y == c]
        rng.shuffle(idx_c)
        n_test = int(round(test_split * len(idx_c)))

        test_idx.extend(idx_c[:n_test].tolist())
        train_idx.extend(idx_c[n_test:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.clip(counts, 1.0, None))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def eval_classifier(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str):
    model.eval()
    loss_sum = 0.0
    n_tot = 0

    ys = []
    preds = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)  # LogSoftmax output
        loss = criterion(out, y)

        loss_sum += float(loss.item()) * x.size(0)
        n_tot += x.size(0)

        pred = out.argmax(1)
        ys.append(y.cpu().numpy())
        preds.append(pred.cpu().numpy())

    ys = np.concatenate(ys)
    preds = np.concatenate(preds)

    avg_loss = loss_sum / max(n_tot, 1)
    acc = accuracy_score(ys, preds)
    return avg_loss, acc, ys, preds


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    class_weights: torch.Tensor | None,
    lr: float,
    epochs: int,
    momentum: float,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if class_weights is not None:
        criterion = nn.NLLLoss(weight=class_weights.to(device))
    else:
        criterion = nn.NLLLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()

    best_test_acc = -1.0
    best_state = None

    for e in range(epochs):
        print(f"\nEpoch {e+1}/{epochs}")

        # ---- TRAIN ----
        model.train()
        loss_meter.reset()
        acc_meter.reset()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            pred = out.argmax(1)
            acc = (pred == y).float().mean().item()

            n = x.size(0)
            loss_meter.add(float(loss.item()), n)
            acc_meter.add(float(acc), n)

        print(f"TRAIN - loss: {loss_meter.value():.4f} | acc: {acc_meter.value():.4f}")

        # ---- TEST ----
        test_loss, test_acc, _, _ = eval_classifier(model, test_loader, criterion, device)
        print(f"TEST  - loss: {test_loss:.4f} | acc: {test_acc:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"--> New best test acc: {best_test_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def load_npz_features(npz_path: Path):
    """
    Supporta 2 formati:
    A) split già pronto:
       train_feats/train_labels/test_feats/test_labels/(val_*)
    B) flat:
       features/labels/class_names
    """
    data = np.load(str(npz_path), allow_pickle=True)
    keys = set(data.keys())

    # Formato A (quello del tuo extract_features_radnet.py)
    if "train_feats" in keys and "train_labels" in keys and "test_feats" in keys and "test_labels" in keys:
        X_train = data["train_feats"]
        y_train = data["train_labels"].astype(np.int64)
        X_test = data["test_feats"]
        y_test = data["test_labels"].astype(np.int64)
        class_names = data["class_names"]
        return X_train, y_train, X_test, y_test, class_names, "split"

    # Formato B (flat)
    if "features" in keys and "labels" in keys and "class_names" in keys:
        X = data["features"]
        y = data["labels"].astype(np.int64)
        class_names = data["class_names"]
        return X, y, None, None, class_names, "flat"

    raise KeyError(
        f"NPZ con chiavi non compatibili. Trovate: {sorted(list(keys))}. "
        "Mi aspetto oppure: "
        "['train_feats','train_labels','test_feats','test_labels','class_names'] "
        "oppure: ['features','labels','class_names']"
    )


def main():
    seed_everything(SEED)

    if not FEATURES_NPZ.exists():
        raise FileNotFoundError(f"Features NPZ non trovato: {FEATURES_NPZ}")

    out = load_npz_features(FEATURES_NPZ)
    mode = out[-1]

    print(f"Feature salvate in: {FEATURES_NPZ}\n")

    if mode == "split":
        X_train, y_train, X_test, y_test, class_names, _ = out
        num_classes = len(class_names)

        print("Loaded:", FEATURES_NPZ)
        print("Train:", X_train.shape, y_train.shape)
        print("Test :", X_test.shape, y_test.shape)
        print("Classes:", list(class_names))

    else:
        # mode == "flat"
        X, y, _, _, class_names, _ = out
        num_classes = len(class_names)

        print("Loaded:", FEATURES_NPZ)
        print("X:", X.shape, "| y:", y.shape)
        print("Classes:", list(class_names))

        X_train, y_train, X_test, y_test = stratified_train_test_split(
            X, y, test_split=TEST_SPLIT, seed=SEED, num_classes=num_classes
        )
        print("\nSplit:")
        print("Train:", X_train.shape, y_train.shape)
        print("Test :", X_test.shape, y_test.shape)

    # Datasets & Loaders
    train_ds = FeaturesDataset(X_train, y_train)
    test_ds = FeaturesDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # class weights
    class_weights = compute_class_weights(y_train, num_classes) if USE_CLASS_WEIGHTS else None
    if class_weights is not None:
        print("Class weights:", class_weights.numpy().round(4).tolist())

    # Model
    model = MLP_Softmax_Classifier(
        in_features=int(X_train.shape[1]),
        hidden_units=256,
        out_classes=num_classes,
    )

    # Train
    model = train_classifier(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        lr=LR,
        epochs=EPOCHS,
        momentum=MOMENTUM,
    )

    # Final evaluation + report
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if class_weights is not None:
        criterion = nn.NLLLoss(weight=class_weights.to(device))
    else:
        criterion = nn.NLLLoss()

    test_loss, test_acc, y_true, y_pred = eval_classifier(model, test_loader, criterion, device)

    print("\n====================")
    print("RISULTATI FINALI")
    print("====================")
    print("Test loss:", f"{test_loss:.4f}")
    print("Accuracy Finale: {:.2f}%".format(test_acc * 100.0))
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[str(c) for c in class_names],
            digits=2,
            zero_division=0,
        )
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print(cm)

    # Save weights
    OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(OUT_WEIGHTS))
    print("\nSaved weights:", OUT_WEIGHTS)


if __name__ == "__main__":
    main()