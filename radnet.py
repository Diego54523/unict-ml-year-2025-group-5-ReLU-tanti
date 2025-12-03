import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms


# ==========================
# CONFIGURAZIONE
# ==========================
DATA_DIR = "archive/MRI"
BATCH_SIZE = 32          # ok così, se la GPU regge puoi provare 64
NUM_EPOCHS = 30          # dai tempo al modello, ci penserà l'early stopping
LR = 3e-5                # più basso: meno salti pazzi, convergenza più stabile
WEIGHT_DECAY = 1e-4      # meno forte di 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
NUM_WORKERS = 8
SEED = 42
EARLY_STOP_PATIENCE = 6  # non killare l’addestramento troppo presto

torch.manual_seed(SEED)
np.random.seed(SEED)


# ==========================
# TRANSFORMS
# ==========================
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # da 1 canale a 3
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # MODIFICA: un filo più forte
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # MODIFICA: piccola variazione intensità
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


# ==========================
# DATASET & DATALOADER
# ==========================
def get_dataloaders(data_dir):
    full_dataset = datasets.ImageFolder(root=data_dir,
                                        transform=train_transform)

    num_classes = len(full_dataset.classes)
    print("Classi trovate:", full_dataset.classes)

    N = len(full_dataset)
    n_test = int(TEST_SPLIT * N)
    n_val = int(VAL_SPLIT * N)
    n_train = N - n_val - n_test

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )

    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform

    # ---- WeightedRandomSampler sul TRAIN ----
    train_targets = [full_dataset.targets[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_targets)
    print("Campioni train per classe:", class_counts)

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_targets]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler=sampler,
                              num_workers=NUM_WORKERS)

    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=NUM_WORKERS)

    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, num_classes, full_dataset.classes


# ==========================
# MODELLO: RADIMAGENET RESNET50
# ==========================
def get_model(num_classes):
    """
    Scarica ResNet50 pre-addestrata su RadImageNet (backbone senza testa)
    e aggiunge una testa di classificazione per il tuo numero di classi.
    """
    backbone = torch.hub.load(
        "Warvito/radimagenet-models",
        model="radimagenet_resnet50",
        verbose=True,
        trust_repo=True,
    )

    class RadImageNetResNet50(nn.Module):
        def __init__(self, backbone, num_classes):
            super().__init__()
            self.backbone = backbone
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(p=0.5)
            self.fc = nn.Linear(2048, num_classes)

        def forward(self, x):
            x = self.backbone(x)          # [B, 2048, H, W]
            x = self.pool(x)              # [B, 2048, 1, 1]
            x = torch.flatten(x, 1)       # [B, 2048]
            x = self.dropout(x)
            x = self.fc(x)
            return x

    model = RadImageNetResNet50(backbone, num_classes)

    # MODIFICA: FREEZING PARZIALE BACKBONE
    # congela tutto il backbone...
    for param in model.backbone.parameters():
        param.requires_grad = False
    # ...ma sblocca l'ultimo blocco (layer4) per un fine-tuning leggero
    if hasattr(model.backbone, "layer4"):
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True

    return model


# ==========================
# LOOP DI TRAINING / VAL
# ==========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_classes):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    per_class_correct = np.zeros(n_classes, dtype=np.int64)
    per_class_total = np.zeros(n_classes, dtype=np.int64)

    for inputs, labels in tqdm(loader, desc="Eval", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)

        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        for c in range(n_classes):
            mask = labels == c
            per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
            per_class_total[c] += mask.sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    recalls = []
    for c in range(n_classes):
        if per_class_total[c] > 0:
            recalls.append(per_class_correct[c] / per_class_total[c])
    balanced_acc = float(np.mean(recalls))

    return epoch_loss, epoch_acc, balanced_acc, per_class_correct, per_class_total


# ==========================
# MAIN
# ==========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, test_loader, num_classes, class_names = \
        get_dataloaders(DATA_DIR)

    model = get_model(num_classes)
    model = model.to(device)

    # MODIFICA: label smoothing per evitare predizioni troppo "rigide"
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # MODIFICA: ottimizziamo solo i parametri con requires_grad=True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    # MODIFICA: scheduler sul learning rate basato sulla balanced accuracy di validazione
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_val_balacc = 0.0
    best_model_path = "best_radimagenet_resnet50.pth"
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== EPOCH {epoch}/{NUM_EPOCHS} =====")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train   - loss: {train_loss:.4f} | acc: {train_acc:.4f}")

        val_loss, val_acc, val_balacc, _, _ = evaluate(
            model, val_loader, criterion, device, num_classes
        )
        print(f"Val     - loss: {val_loss:.4f} | acc: {val_acc:.4f} | bal_acc: {val_balacc:.4f}")

        # step del scheduler con la metrica di interesse
        scheduler.step(val_balacc)

        # salva il modello migliore in base alla balanced accuracy
        if val_balacc > best_val_balacc:
            best_val_balacc = val_balacc
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Nuovo best model salvato ({best_model_path})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Nessun miglioramento per {epochs_no_improve} epoche.")

        # MODIFICA: early stopping
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping: la balanced accuracy non migliora più.")
            break

    # =============== TEST FINALE ===============
    print("\nCarico il modello migliore per il test...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, test_balacc, per_class_correct, per_class_total = evaluate(
        model, test_loader, criterion, device, num_classes
    )

    print("\n===== RISULTATI TEST =====")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Balanced Accuracy: {test_balacc:.4f}")

    print("\nAccuracy per classe:")
    for idx, cls in enumerate(class_names):
        if per_class_total[idx] > 0:
            acc_cls = per_class_correct[idx] / per_class_total[idx]
            print(f"{cls}: {acc_cls:.4f}  ({per_class_correct[idx]}/{per_class_total[idx]})")


if __name__ == "__main__":
    main()