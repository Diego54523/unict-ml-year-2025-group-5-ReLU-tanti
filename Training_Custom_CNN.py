import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from dotenv import load_dotenv
from Get_Mean_std import get_mean_std
from Custom_CNN_for_Feature_Extraction import CustomCNNFeatureExtractor
from Func_train_CNN import train_classifier
from torchvision.datasets import ImageFolder
import gc #Importo questa libreria per la gestione della memoria, dato che anche con una BATCH di 32 arivati al layer3 la memoria GPU si esaurisce, con questa classe per sicurezza pulisco innanzitutto la memoria prima di addestrare il modello
import time

#Creo una classe per caricare tutto in memoria in modo da ridurre di molto i tempi eliminando l'accesso a memoria ogni volta che chiedo un batch
class InMemoryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        print(f"Caricamento dataset in RAM da: {root_dir}...")
        start_t = time.time()
        
        # Usiamo ImageFolder temporaneo per leggere dal disco
        temp_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        
        self.data = []
        self.targets = temp_dataset.targets # Copiamo le label
        
        # Ciclo di caricamento
        for i in range(len(temp_dataset)):
            img, label = temp_dataset[i]
            self.data.append((img, label))
            
            # Stampa progresso ogni 1000 immagini
            if (i + 1) % 1000 == 0:
                print(f"  Caricate {i + 1}/{len(temp_dataset)} immagini...")
                
        end_t = time.time()
        print(f"Finito! Dataset caricato in RAM in {end_t - start_t:.1f} secondi.")

    def __getitem__(self, index):
        # Ritorna l'immagine già in memoria (velocissimo)
        return self.data[index]

    def __len__(self):
        return len(self.data)

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
    counts = np.bincount(labels) # Conta il numero di occorrenze per ogni classe

    # Pesi inversi: 1 / count
    weights = 1. / counts
    # Normalizziamo i pesi
    weights = weights / weights.sum() #Fa in modo che la somma dei pesi sia 1, in modo da non creare problemi alla funzione di Loss con pesi enormi

    return torch.FloatTensor(weights).to(device)


if __name__ == '__main__':
    load_dotenv()
    
    # Configurazione Parametri
    dataSetPath = os.getenv("DATA_PATH")
    if dataSetPath is None:
        raise ValueError("DATA_PATH non definito nel file .env")
    
    #Pulisco la memoria GPU prima di addestrare il modello
    torch.cuda.empty_cache()
    gc.collect()
        
    BATCH_SIZE = 16
    NUM_WORKERS = 4 #Per alleggerire il carico sulla memoria RAM, imposto a 0 il numero di workers (su Windows spesso dà problemi con valori >0)
    IMG_SIZE = (178, 208) 
    DIVIDER = 0.8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in uso: {DEVICE}")

    # Caricamento Dataset BASE
    # Manteniamo le immagini come PIL per permettere una Data Augmentation migliore dopo
    raw_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor() # Convertiamo subito in tensore per la RAM
    ])

    full_dataset = ImageFolder(root=dataSetPath, transform=raw_transform)

    # Split Training e Validation
    trainSize = int(DIVIDER * len(full_dataset))
    valSize = len(full_dataset) - trainSize

    # Nota: Questi subset puntano al dataset con 'raw_transform' (immagini PIL)
    train_subset_raw, val_subset_raw = random_split(
        full_dataset, 
        [trainSize, valSize],
        generator=torch.Generator().manual_seed(42)
    )

    # Calcolo Mean e Std (Usiamo un DataLoader temporaneo sul Training Set)
    print("Calcolo Mean e Std sul training set...")

    calc_transform = transforms.Compose([])

    calc_dataset = TransformedSubset(train_subset_raw, transform=calc_transform)
    calc_loader = DataLoader(calc_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    mean, std = get_mean_std(calc_loader)
    print(f'Calculated Mean: {mean}')
    print(f'Calculated Std: {std}')

    #Trasformazioni Finali
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),      # Data Augmentation (su PIL)
        transforms.RandomRotation(15),  # Data Augmentation (su PIL)
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Data Augmentation (su PIL)
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Data Augmentation (su PIL)
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()) # Normalizzazione
    ])

    val_transform = transforms.Compose([
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    # Creazione Dataset Finali usando il Wrapper
    final_train_dataset = TransformedSubset(train_subset_raw, transform=train_transform)
    final_val_dataset = TransformedSubset(val_subset_raw, transform=val_transform)

    # Creazione DataLoader Finali
    trainLoader = DataLoader(final_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valLoader = DataLoader(final_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Calcolo Class Weights
    try:
        class_weights = calculate_class_weights(full_dataset, train_subset_raw.indices, DEVICE)
        print(f"Class Weights calcolati: {class_weights}")
    except Exception as e:
        print(f"Attenzione: Impossibile calcolare i pesi delle classi ({e}). Uso None.")
        class_weights = None

    # Setup Modello
    model = CustomCNNFeatureExtractor(num_classes=4)
    model = model.to(DEVICE)

    print("Avvio training...")
    model = train_classifier(
        model, 
        train_loader=trainLoader, 
        test_loader=valLoader, 
        class_weights=class_weights, 
        exp_name="custom_cnn_classifier", 
        logdir="logs_custom_cnn_classifier"
    )