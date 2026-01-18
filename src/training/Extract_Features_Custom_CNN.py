import torch
import torch.nn as nn
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from dotenv import load_dotenv
from models.Custom_CNN import CustomCNNFeatureExtractor

def extract_features_to_numpy(model, loader, device):
    model.eval() # Fondamentale: disattiviamo Dropout e BatchNorm
    features_list = []
    labels_list = []

    with torch.no_grad(): # Risparmio di memoria e calcoli
        for images, labels in tqdm(loader, desc = "Estraendo features"):
            images = images.to(device)
            
            # Passiamo l'immagine nella CNN. 
            # Dato che abbiamo modificato l'ultimo layer, 
            # l'output sarà il vettore delle feature (es. 128 numeri)
            output = model(images)
            
            # Spostiamo su CPU e convertiamo in Numpy
            features_list.append(output.cpu().numpy())
            labels_list.append(labels.numpy())

    # Concateniamo tutti i batch
    features = np.concatenate(features_list, axis = 0)
    labels = np.concatenate(labels_list, axis = 0)
    return features, labels

def main():
    load_dotenv()
    
    dataSetPath = os.getenv("DATA_PATH")
    if dataSetPath is None:
        print("Errore: DATA_PATH non trovato nel file .env")
        sys.exit()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "weights", "custom_cnn_classifier_BEST.pth") 

    output_filename = "mri_features_custom.npz"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = (178, 208) 
    BATCH_SIZE = 64 # Possiamo usare batch più grandi perché non facciamo training

    print(f"Device: {DEVICE}")

    # IMPORTANTE: Usiamo SOLO la trasformazione di "Validazione" (senza rotazioni/flip).
    # Vogliamo estrarre le caratteristiche pure dell'immagine.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels = 1), # 1 Canale come da training
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.2954], std = [0.3207]) 
    ])

    full_dataset = datasets.ImageFolder(root = dataSetPath, transform = transform)
    classes = full_dataset.classes
    
    # Ricostruiamo lo split 80/20 identico al training (seed 42)
    trainSize = int(0.8 * len(full_dataset))
    valSize = len(full_dataset) - trainSize
    
    train_subset, val_subset = random_split(
        full_dataset, [trainSize, valSize], generator = torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
    val_loader = DataLoader(val_subset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)

    # PREPARAZIONE MODELLO
    print("Caricamento CNN e Pesi...")
    model = CustomCNNFeatureExtractor(num_classes = 4) 
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location = DEVICE))
        print("Pesi caricati correttamente.")
    else:
        print(f"ERRORE CRITICO: Non trovo i pesi in {weights_path}")
        sys.exit()

    # Fase di decapitazione
    # Sostituiamo l'ultimo layer (Linear) con Identity().
    # In questo modo la rete non fa più la classificazione finale, 
    # ma restituisce direttamente l'input che arrivava al layer lineare.
    model.linear = nn.Identity()
    
    model.to(DEVICE)

    # ESTRAZIONE FEATURES 
    print("\nInizio estrazione features Training Set...")
    X_train, y_train = extract_features_to_numpy(model, train_loader, DEVICE)

    print("\nInizio estrazione features Validation Set...")
    X_val, y_val = extract_features_to_numpy(model, val_loader, DEVICE)

    print(f"\nDimensioni Train Features: {X_train.shape}")
    print(f"Dimensioni Val Features: {X_val.shape}")

    #SALVATAGGIO
    save_path = os.path.join(script_dir, output_filename)
    np.savez(save_path, 
             train_feats = X_train, 
             train_labels = y_train, 
             val_feats = X_val, 
             val_labels = y_val,
             classes = classes)
    
    print(f"\nFeatures salvate in: {save_path}")
    
    
if __name__ == "__main__":
    main()