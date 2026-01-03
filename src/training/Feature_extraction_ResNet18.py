from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torch
import numpy as np
from torchvision import models
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Su Windows non posso usare più threads per caricare il dataset quindi devo definire manualmente il processo padre per chiamare i threads.
if __name__ == '__main__':
    load_dotenv()

    dataSetPath = os.getenv("DATA_PATH")

    # Preprocess
    transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels = 3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                            std = [0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root = dataSetPath, transform = transformer)
    batchSize = 32
    shuffle = True
    numWorkers = 4
    divider = 0.8

    trainSize = int(divider * len(dataset))
    valSize = len(dataset) - trainSize

    trainDataset, valDataset = random_split(
        dataset, 
        [trainSize, valSize],
        generator = torch.Generator().manual_seed(42)
    )

    trainLoader = DataLoader(
        trainDataset, 
        batch_size = batchSize, 
        shuffle = shuffle, 
        num_workers = numWorkers
    )

    valLoader = DataLoader(
        valDataset, 
        batch_size = batchSize, 
        shuffle = shuffle, 
        num_workers = numWorkers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

    featureExtractor = torch.nn.Sequential(
        *list(model.children())[:-1]).to(device).eval()


    batch = next(iter(trainLoader))
    imgs, labels = batch 

    with torch.no_grad():
        feats = featureExtractor(imgs.to(device))
        feats = feats.view(feats.size(0), -1)

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

    train_feats, train_labels = extract_embeddings(
        trainLoader, featureExtractor, device)

    te_feats, te_labels = extract_embeddings(
        valLoader, featureExtractor, device)

    classes = dataset.classes

    script_dir = os.path.dirname(os.path.abspath(__file__))

    output_path = os.path.join(script_dir, "mri_features.npz")

    np.savez_compressed(output_path,
        train_feats = train_feats,
        train_labels = train_labels,
        val_feats = te_feats,
        val_labels = te_labels,
        classes = np.array(classes)
    )