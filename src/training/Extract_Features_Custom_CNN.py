import torch
import torch.nn as nn
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
from models.Custom_CNN import CustomCNNFeatureExtractor

def extract_features_to_numpy(model, loader, device):
    model.eval() # Fondamentale: disattiviamo Dropout e BatchNorm
    features_list = []
    labels_list = []

    with torch.no_grad(): # Risparmio di memoria e calcoli
        for images, labels in tqdm(loader, desc = "Estraendo features"):
            images = images.to(device)            
            output = model(images)
            
            features_list.append(output.cpu().numpy())
            labels_list.append(labels.numpy())

    features = np.concatenate(features_list, axis = 0)
    labels = np.concatenate(labels_list, axis = 0)
    return features, labels

def main():
    load_dotenv()
    dataSetPath = os.getenv("DATA_PATH")
    
    if dataSetPath is None:
        print("Errore: DATA_PATH non trovato nel file .env")
        sys.exit()

    # ==========================
    # PROJECT PATHS
    # ==========================
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    RESULTS_DIR = PROJECT_ROOT / "results"
    WEIGHTS_DIR = RESULTS_DIR / "weights"
    FEATURES_DIR = RESULTS_DIR / "features"

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    weights_path = WEIGHTS_DIR / "custom_cnn_classifier_BEST.pth"
    save_path = FEATURES_DIR / "mri_features_custom.npz"
    
    # ==========================
    # CONFIG
    # ==========================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = (178, 208) 
    BATCH_SIZE = 64 

    print(f"Device: {DEVICE}")

    # ==========================
    # TRANSFORMS (VALIDATION ONLY)
    # ==========================
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels = 1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.2954], std = [0.3207]) 
    ])

    # ==========================
    # DATASET & SPLIT
    # ==========================
    full_dataset = datasets.ImageFolder(root = dataSetPath, transform = transform)
    classes = full_dataset.classes
    
    trainSize = int(0.8 * len(full_dataset))
    valSize = len(full_dataset) - trainSize
    
    train_subset, val_subset = random_split(
        full_dataset, [trainSize, valSize], generator = torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
    val_loader = DataLoader(val_subset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)

    # ==========================
    # MODEL
    # ==========================
    print("Caricamento CNN e Pesi...")
    model = CustomCNNFeatureExtractor(num_classes = 4) 
    
    if not weights_path.exists():
        print(f"ERRORE CRITICO: checkpoint non trovato in {weights_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    print("Pesi caricati correttamente.")

    # fase di decapitazione
    model.linear = nn.Identity()
    model.to(DEVICE)

    # ==========================
    # FEATURE EXTRACTION
    # ==========================
    print("\nInizio estrazione features Training Set...")
    X_train, y_train = extract_features_to_numpy(model, train_loader, DEVICE)

    print("\nInizio estrazione features Validation Set...")
    X_val, y_val = extract_features_to_numpy(model, val_loader, DEVICE)

    print(f"\nDimensioni Train Features: {X_train.shape}")
    print(f"Dimensioni Val Features: {X_val.shape}")

    # ==========================
    # SAVE
    # ==========================
    np.savez_compressed(
        save_path,
        train_feats=X_train,
        train_labels=y_train,
        val_feats=X_val,
        val_labels=y_val,
        classes=np.array(classes),
    )

    print(f"\nFeature salvate correttamente in:\n{save_path}")
    
    
if __name__ == "__main__":
    main()