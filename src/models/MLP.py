from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from src.models.MLP_Softmax_Class import MLP_Softmax_Classifier
from src.training.Func_Softmax_MLP_train_classifier import train_classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))

# Carichiamo le feature estratte e salvate in precedenza
data = np.load(os.path.join(script_dir, "mri_features_custom.npz"))
X_train = data['train_feats']
y_train = data['train_labels']
X_val = data['val_feats']
y_val = data['val_labels']

classes = data['classes'].tolist()

# MLP classifier

# Convertiamo i numpy array (le feature estratte da ResNet) in tensori PyTorch
feat_train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
feat_val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

# Creiamo dei nuovi DataLoader per features (vettori lunghi 512) invece di immagini
feat_train_loader = DataLoader(feat_train_data, batch_size = 32, shuffle = True)
feat_val_loader = DataLoader(feat_val_data, batch_size = 32, shuffle = False)

# CONFIGURIAMO L'MLP PER LE FEATURE (INPUT_SIZE = 512 e HIDDEN_UNITS = 256 per ResNet18, ora rispettivamente 128 e 64 per Custom CNN)
INPUT_SIZE = 128
HIDDEN_UNITS = 64
NUM_CLASSES = 4

# Calcoliamo dei pesi che verranno usati per  alla frequenza (più è rara la classe, più alto il peso)
# y_train contiene le etichette numeriche (0, 1, 2, 3)
class_weights = compute_class_weight(
    class_weight = 'balanced', 
    classes = np.unique(y_train), 
    y = y_train
)
    
# Convertiamo in tensore PyTorch e spostiamo su GPU/CPU
class_weights = torch.tensor(class_weights, dtype = torch.float).to(device)
    
# Esempio output atteso: [1.0, 10.5, 1.2, 1.1] -> La classe con pochi samples ha peso altissimo

# ADDESTRAMENTO
mri_mlp_classifier = MLP_Softmax_Classifier(in_features = INPUT_SIZE, hidden_units = HIDDEN_UNITS, out_classes = NUM_CLASSES)

mri_mlp_classifier = train_classifier(mri_mlp_classifier, train_loader = feat_train_loader, test_loader = feat_val_loader, class_weights = class_weights, exp_name = "mri_mlp_classifier", logdir = "logs_mri_mlp_classifier")

# VALUTAZIONE FINALE
mri_mlp_classifier.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    # Usiamo feat_val_loader perché contiene già i vettori pronti [Batch, 512]
    for batch in tqdm(feat_val_loader, desc = "Valutazione"):
        features, labels = batch
        features = features.to(device)
            
        # Non serve .view(), le feature sono già piatte (512)
        outputs = mri_mlp_classifier(features)
        _, max_index = torch.max(outputs, 1)

        all_preds.extend(max_index.cpu().numpy())
        all_labels.extend(labels.numpy())

val_accuracy = accuracy_score(all_labels, all_preds)
print(f"\nAccuracy Finale: {val_accuracy * 100:.2f}%")
print(classification_report(all_labels, all_preds, target_names = classes))
print(confusion_matrix(all_labels, all_preds))
