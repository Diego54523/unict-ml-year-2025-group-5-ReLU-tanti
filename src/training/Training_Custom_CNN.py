import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from dotenv import load_dotenv
from src.data.Get_Mean_std import get_mean_std
from src.models.Custom_CNN import CustomCNNFeatureExtractor
from src.training.Func_train_CNN import train_classifier
from torchvision.datasets import ImageFolder
import gc #Importo questa libreria per la gestione della memoria, dato che anche con una BATCH di 32 arivati al layer3 la memoria GPU si esaurisce, con questa classe per sicurezza pulisco innanzitutto la memoria prima di addestrare il modello
from sklearn.metrics import classification_report
from torch.utils.data import WeightedRandomSampler

class TransformedSubset(Dataset):
    """
    Wrapper che permette di applicare trasformazioni diverse (es. Augmentation vs Normalize)
    ai subset creati da random_split.
    """
    def __init__(self, subset, transform = None):
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
    
    # Pulisco la memoria GPU prima di addestrare il modello
    torch.cuda.empty_cache()
    gc.collect()
        
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    IMG_SIZE = (178, 208) 
    DIVIDER = 0.8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in uso: {DEVICE}")

    # Caricamento Dataset BASE
    # Manteniamo le immagini come PIL per permettere una Data Augmentation migliore dopo
    raw_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels = 1),
        transforms.Resize(IMG_SIZE)
    ])

    full_dataset = ImageFolder(root = dataSetPath, transform = raw_transform)

    # Split Training e Validation
    trainSize = int(DIVIDER * len(full_dataset))
    valSize = len(full_dataset) - trainSize

    # Nota: Questi subset puntano al dataset con 'raw_transform' (immagini PIL)
    train_subset_raw, val_subset_raw = random_split(
        full_dataset, 
        [trainSize, valSize],
        generator = torch.Generator().manual_seed(42)
    )

    # Calcolo Mean e Std (Usiamo un DataLoader temporaneo sul Training Set)
    print("Calcolo Mean e Std sul training set...")

    calc_transform = transforms.Compose([transforms.ToTensor()])  # Solo ToTensor per calcolo mean/std

    calc_dataset = TransformedSubset(train_subset_raw, transform = calc_transform)
    calc_loader = DataLoader(calc_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)
    
    mean, std = get_mean_std(calc_loader)
    print(f'Calculated Mean: {mean}')
    print(f'Calculated Std: {std}')

    #Trasformazioni Finali
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p = 0.5),      # Data Augmentation (su PIL)
        transforms.RandomRotation(15),  # Data Augmentation (su PIL)
        transforms.RandomAffine(degrees = 0, translate = (0.15, 0.15), scale = (0.9, 1.1)), # Data Augmentation (su PIL)
        transforms.ToTensor(),
        transforms.Normalize(mean = mean.tolist(), std = std.tolist()) # Normalizzazione
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = mean.tolist(), std = std.tolist())
    ])

    print("Configurazione WeightedRandomSampler...") # Serve per far si che in ogni batch ci sia lo stesso numero di esempi per ogni classe, in modo da bilanciare il training
    train_targets = [full_dataset.targets[i] for i in train_subset_raw.indices]
    class_counts = np.bincount(train_targets)
    class_weights_calc = 1. / class_counts
    sample_weights = [class_weights_calc[t] for t in train_targets]
    
    sampler = WeightedRandomSampler(
        weights = sample_weights, 
        num_samples = len(sample_weights), 
        replacement = True
    )

    # Creazione Dataset Finali usando il Wrapper
    final_train_dataset = TransformedSubset(train_subset_raw, transform = train_transform)
    final_val_dataset = TransformedSubset(val_subset_raw, transform = val_transform)

    # Creazione DataLoader Finali
    trainLoader = DataLoader(final_train_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS, sampler = sampler)
    valLoader = DataLoader(final_val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS)

    # Setup Modello
    model = CustomCNNFeatureExtractor(num_classes = 4)
    model = model.to(DEVICE)

    print("Avvio training...")
    model = train_classifier(
        model, 
        train_loader = trainLoader, 
        test_loader = valLoader, 
        class_weights = None, 
        exp_name = "custom_cnn_classifier", 
        logdir = "logs_custom_cnn_classifier",
        patience = 5
    )

    model.eval() # Modalità valutazione
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in valLoader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_acc = 100 * correct / total
    print(f'\nAccuracy Finale del Miglior Modello sul Validation Set: {final_acc:.2f}%')

    # Opzionale: Matrice di Confusione
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names = ["Mild", "Moderate", "Non", "Very Mild"]))