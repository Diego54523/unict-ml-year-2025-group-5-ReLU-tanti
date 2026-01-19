from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torch
import numpy as np
from torchvision import models
from tqdm import tqdm
import os
from pathlib import Path
from dotenv import load_dotenv

def extract_embeddings(dataloader, feature_extractor, device):
        all_feats = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                imgs, labels = batch
                imgs = imgs.to(device)

                feats = feature_extractor(imgs)
                feats = feats.view(feats.size(0), -1)

                all_feats.append(feats.cpu())
                all_labels.append(labels)

        all_feats = torch.cat(all_feats, dim = 0)
        all_labels = torch.cat(all_labels, dim = 0)

        return all_feats.numpy(), all_labels.numpy()

def main():
    load_dotenv()
    dataSetPath = os.getenv("DATA_PATH")
    
    if not dataSetPath:
        raise SystemExit("DATA_PATH non impostata. Mettila nel .env (DATA_PATH=...)")
    
    # ==========================
    # PROJECT PATHS
    # ==========================
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ML-Project/
    FEATURES_DIR = PROJECT_ROOT / "results" / "features"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FEATURES_DIR / "mri_features.npz"
    
    # ==========================
    # CONFIG
    # ==========================
    batchSize = 32
    numWorkers = 0 if os.name == "nt" else 4
    divider = 0.8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================
    # PREPROCESS
    # ==========================
    transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])
    

    dataset = ImageFolder(root=dataSetPath, transform=transformer)
    classes = dataset.classes

    trainSize = int(divider * len(dataset))
    valSize = len(dataset) - trainSize

    trainDataset, valDataset = random_split(
        dataset,
        [trainSize, valSize],
        generator=torch.Generator().manual_seed(42),
    )

    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=numWorkers,
    )

    valLoader = DataLoader(
        valDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
    )
    
    # ==========================
    # MODEL (FEATURE EXTRACTOR)
    # ==========================
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    featureExtractor = torch.nn.Sequential(
        *list(model.children())[:-1]
    ).to(device).eval()

    # ==========================
    # EXTRACTION
    # ==========================
    train_feats, train_labels = extract_embeddings(trainLoader, featureExtractor, device)
    val_feats, val_labels = extract_embeddings(valLoader, featureExtractor, device)

    # ==========================
    # SAVE
    # ==========================
    np.savez_compressed(
        output_path,
        train_feats=train_feats,
        train_labels=train_labels,
        val_feats=val_feats,
        val_labels=val_labels,
        classes=np.array(classes),
    )

    print(f"\nFeatures salvate correttamente in:\n{output_path}")
    
if __name__ == "__main__":
    main()