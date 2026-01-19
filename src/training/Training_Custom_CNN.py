import os
import gc
from pathlib import Path

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from dotenv import load_dotenv
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report
from torch.utils.data import WeightedRandomSampler

from data.Get_Mean_std import get_mean_std
from models.Custom_CNN import CustomCNNFeatureExtractor
from training.Func_train_CNN import train_classifier

class TransformedSubset(Dataset):
    """
    Wrapper che permette di applicare trasformazioni diverse (es. Augmentation vs Normalize)
    ai subset creati da random_split.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def calculate_class_weights(dataset, subset_indices, device):
    """Calcola i pesi per gestire lo sbilanciamento delle classi"""
    labels = [dataset.targets[i] for i in subset_indices]
    counts = np.bincount(labels)

    weights = 1.0 / counts
    weights = weights / weights.sum()

    return torch.FloatTensor(weights).to(device)


def main():
    load_dotenv()

    # ==========================
    # DATASET PATH
    # ==========================
    dataSetPath = os.getenv("DATA_PATH")
    if not dataSetPath:
        raise ValueError("DATA_PATH non definito nel file .env")

    # ==========================
    # PROJECT PATHS (results/)
    # ==========================
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ML-Project/
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_DIR = RESULTS_DIR / "logs"
    REPORTS_DIR = RESULTS_DIR / "reports"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================
    # MEMORY CLEANUP
    # ==========================
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ==========================
    # CONFIG
    # ==========================
    BATCH_SIZE = 16
    IMG_SIZE = (178, 208)
    DIVIDER = 0.8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0 if os.name == "nt" else 4

    print(f"Device in uso: {DEVICE}")
    print(f"Dataset: {dataSetPath}")

    # ==========================
    # BASE DATASET (PIL)
    # ==========================
    raw_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
    ])

    full_dataset = ImageFolder(root=dataSetPath, transform=raw_transform)

    # ==========================
    # SPLIT TRAIN/VAL
    # ==========================
    trainSize = int(DIVIDER * len(full_dataset))
    valSize = len(full_dataset) - trainSize

    train_subset_raw, val_subset_raw = random_split(
        full_dataset,
        [trainSize, valSize],
        generator=torch.Generator().manual_seed(42),
    )

    # ==========================
    # MEAN/STD (train only)
    # ==========================
    print("Calcolo Mean e Std sul training set...")
    calc_transform = transforms.Compose([transforms.ToTensor()])
    calc_dataset = TransformedSubset(train_subset_raw, transform=calc_transform)
    calc_loader = DataLoader(
        calc_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    mean, std = get_mean_std(calc_loader)
    print(f"Calculated Mean: {mean}")
    print(f"Calculated Std:  {std}")

    # ==========================
    # TRANSFORMS FINALI
    # ==========================
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    # ==========================
    # WEIGHTED SAMPLER
    # ==========================
    print("Configurazione WeightedRandomSampler...")
    train_targets = [full_dataset.targets[i] for i in train_subset_raw.indices]
    class_counts = np.bincount(train_targets)
    class_weights_calc = 1.0 / class_counts
    sample_weights = [class_weights_calc[t] for t in train_targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # ==========================
    # FINAL DATASETS/LOADERS
    # ==========================
    final_train_dataset = TransformedSubset(train_subset_raw, transform=train_transform)
    final_val_dataset = TransformedSubset(val_subset_raw, transform=val_transform)

    trainLoader = DataLoader(
        final_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,           
        num_workers=NUM_WORKERS,
        sampler=sampler,
    )

    valLoader = DataLoader(
        final_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # ==========================
    # MODEL
    # ==========================
    model = CustomCNNFeatureExtractor(num_classes=4).to(DEVICE)

    # ==========================
    # TRAIN
    # ==========================
    print("Avvio training...")
    exp_name = "custom_cnn_classifier"

    model = train_classifier(
        model,
        train_loader=trainLoader,
        test_loader=valLoader,
        class_weights=None,
        exp_name=exp_name,
        logdir=str(LOGS_DIR), 
        patience=5,
    )

    # ==========================
    # FINAL EVAL (val)
    # ==========================
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in valLoader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Report stampato + salvato su file
    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=["Mild", "Moderate", "Non", "Very Mild"],
    )

    print("\nClassification Report:")
    print(report_text)

    report_path = REPORTS_DIR / f"{exp_name}_classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nReport salvato in:\n{report_path}")
    
    
if __name__ == "__main__":
    main()