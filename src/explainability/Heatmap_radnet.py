# src/training/heatmap_radnet50.py

import os
import sys
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv


# ==========================
# ENV + PATH
# ==========================
load_dotenv()
dataSetPath = os.getenv("DATA_PATH")

if dataSetPath is None:
    raise RuntimeError("DATA_PATH non trovato nel .env")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# TRANSFORMS (coerenti con feature extraction)
# ==========================
transformer = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ==========================
# DATASET / LOADER (val split 80/20)
# ==========================
try:
    dataset = ImageFolder(root=dataSetPath, transform=transformer)
    classes = dataset.classes
except FileNotFoundError:
    print(f"ERRORE: Non trovo il dataset in: {dataSetPath}")
    print("Devi scaricare le immagini o correggere il percorso.")
    raise

trainSize = int(0.8 * len(dataset))
valSize = len(dataset) - trainSize
_, valDataset = random_split(dataset, [trainSize, valSize], generator=torch.Generator().manual_seed(42))

# IMPORTANTE: per evitare problemi multiprocessing su Fedora/Python 3.14
# se ti dà errori, metti num_workers=0.
valLoader = DataLoader(valDataset, batch_size=32, shuffle=True, num_workers=0)


# ==========================
# LOAD RADNET50 (RadImageNet)
# ==========================
# Repo torch.hub usato da te: Warvito/radimagenet-models
# N.B. in quel repo spesso NON c'è argomento pretrained=True, quindi non passarlo.
# Se il nome entrypoint è diverso, cambia 'radimagenet_resnet50' con quello corretto.
model = torch.hub.load(
    "Warvito/radimagenet-models",
    "radimagenet_resnet50",
    trust_repo=True,
)
model = model.to(device)
model.eval()


# ==========================
# HOOK su layer4 (ultima feature map conv)
# ==========================
activations = None

def get_activation_hook(module, input, output):
    global activations
    activations = output.detach()

# ResNet-like: layer4 esiste
handle = model.layer4.register_forward_hook(get_activation_hook)


# ==========================
# Denormalize per visualizzazione
# ==========================
def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = tensor.permute(1, 2, 0).detach().cpu().numpy()  # (H,W,C)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


# ==========================
# Heatmap logic
# ==========================
def make_heatmap_from_featuremap(fmap_2d: np.ndarray):
    """
    fmap_2d: (H_small, W_small) già ottenuta facendo mean sui canali.
    """
    heatmap = np.maximum(fmap_2d, 0)  # ReLU
    mx = float(np.max(heatmap))
    if mx > 0:
        heatmap = heatmap / mx
    return heatmap


# ==========================
# GENERAZIONE HEATMAP
# ==========================
TOTAL_IMAGES = 80
IMAGES_PER_PAGE = 8

images_processed = 0

try:
    for images_batch, labels_batch in tqdm(valLoader, desc="Heatmaps"):
        if images_processed >= TOTAL_IMAGES:
            break

        images_batch = images_batch.to(device, non_blocking=True)

        # forward -> attiva hook
        _ = model(images_batch)

        if activations is None:
            print("ERRORE: activations None (hook non ha catturato nulla).")
            break

        # atteso per ResNet50: [B, 2048, 7, 7]
        current_maps = activations

        for i in range(len(images_batch)):
            if images_processed >= TOTAL_IMAGES:
                break

            # nuova pagina
            if images_processed % IMAGES_PER_PAGE == 0:
                plt.show()
                plt.figure(figsize=(20, 10))
                print(f"--- Gruppo {images_processed // IMAGES_PER_PAGE + 1} ---")

            # mean sui canali -> (H_small, W_small)
            fm = current_maps[i].mean(dim=0).detach().cpu().numpy()
            heatmap = make_heatmap_from_featuremap(fm)

            img_original = denormalize(images_batch[i])  # float [0,1], (H,W,C)
            img_uint8 = np.uint8(255 * img_original)

            # heatmap resized su (W,H)
            heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            superimposed = np.uint8(0.6 * img_uint8 + 0.4 * heatmap_colored)
            combined_view = np.hstack((img_uint8, superimposed))

            # plot
            idx_in_page = (images_processed % IMAGES_PER_PAGE) + 1
            plt.subplot(2, 4, idx_in_page)
            plt.imshow(combined_view)
            plt.title(classes[int(labels_batch[i].item())])
            plt.axis("off")

            images_processed += 1

        plt.tight_layout()

finally:
    plt.show()
    handle.remove()
