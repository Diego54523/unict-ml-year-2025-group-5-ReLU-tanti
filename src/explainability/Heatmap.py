import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
dataSetPath = os.getenv("DATA_PATH")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dobbiamo ricreare il loader identico a come è stato fatto nel training
transformer = transforms.Compose([
    transforms.Grayscale(num_output_channels = 3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])
])

# Carichiamo il dataset, lo useremo solo per visualizzare le immagini di validazione
try:
    dataset = ImageFolder(root = dataSetPath, transform = transformer)
    classes = dataset.classes
except FileNotFoundError:
    print(f"ERRORE: Non trovo il dataset in: {dataSetPath}")
    print("Devi scaricare le immagini o correggere il percorso.")
    exit()


trainSize = int(0.8 * len(dataset))
valSize = len(dataset) - trainSize
_, valDataset = random_split(dataset, [trainSize, valSize], generator = torch.Generator().manual_seed(42))

valLoader = DataLoader(valDataset, batch_size = 32, shuffle = True)

# Usiamo ResNet18 pre-allenata (come feature extractor)
model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
model = model.to(device)
model.eval()

# PREPARAZIONE HEATMAP 
activations = None
def get_activation_hook(model, input, output):
    global activations # Serve per indicare alla funzione che non deve creare una variabile locale per salvare i dati ma usare quella globale.
    activations = output.detach()

# Registriamo l'hook
handle = model.layer4.register_forward_hook(get_activation_hook)

def denormalize(tensor): # Le denormalizza per poterle visualizzare correttamente
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy() # Cambia da [Channels,Height,Width], standard di pytorch per le immagini, a [H,W,C]
    img = std * img + mean # Formula inversa della normalizzazione
    img = np.clip(img, 0, 1) # Ci assicuriamo che i valori siano tra 0 e 1
    return img

# GENERAZIONE HEATMAP
TOTAL_IMAGES = 80
IMAGES_PER_PAGE = 8

images_processed = 0

# Iteriamo sul DataLoader
for batch in valLoader:
    if images_processed >= TOTAL_IMAGES: break
    
    images_batch, labels_batch = batch
    images_batch = images_batch.to(device)
    output = model(images_batch) # l'hook si attiva automaticamente qui
    
    current_maps = activations # [Batch, 512, 7, 7] 
    
    # Processiamo il batch corrente
    for i in range(len(images_batch)):
        if images_processed >= TOTAL_IMAGES: break
        
        # Se è l'inizio di una nuova "pagina", crea una nuova figura
        if images_processed % IMAGES_PER_PAGE == 0:
            plt.show() # Mostra la vecchia figura
            plt.figure(figsize = (20, 10)) # Crea nuova tela
            print(f"--- Gruppo {images_processed // IMAGES_PER_PAGE + 1} ---")

        # Logica Heatmap (uguale a prima)
        fm = current_maps[i].mean(dim = 0).cpu().numpy() # Media su tutti i canali, che ci dice dove l'attivazione c'è più attivita in generale
        heatmap = np.maximum(fm, 0) # ReLU manuale
        if np.max(heatmap) != 0: heatmap /= np.max(heatmap) # Normalizza l'heatmap tra 0 e 1
        
        img_original = denormalize(images_batch[i])
        img_original_uint8 = np.uint8(255 * img_original) # Convertiamo in uint8 per OpenCV, in modo che i valori siano tra 0 e 255
        heatmap_resized = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0])) 
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET) # La scala di colori andrà da blu (bassa attenzione) a rosso (alta attenzione)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # Convertiamo da BGR a RGB, perché OpenCV usa BGR di default mentre matplotlib usa RGB
        
        superimposed = np.uint8(0.6 * img_original_uint8 + 0.4 * heatmap_colored)
        combined_view = np.hstack((img_original_uint8, superimposed)) # Horizontal stack: immagine originale a sinistra, heatmap a destra
        
        # Plot 
        idx_in_page = (images_processed % IMAGES_PER_PAGE) + 1
        ax = plt.subplot(2, 4, idx_in_page) # 2 righe, 4 colonne
        plt.imshow(combined_view)
        plt.title(classes[labels_batch[i].item()])
        plt.axis('off')
        
        images_processed += 1
    
    plt.tight_layout()

plt.show()
handle.remove()