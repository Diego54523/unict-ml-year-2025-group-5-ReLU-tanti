import torch
import numpy as np
from tqdm import tqdm
from models.MLP_Softmax_Class import MLP_Softmax_Classifier
from training.Func_Softmax_MLP_train_classifier import train_classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================
    # PROJECT PATHS
    # ==========================
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ML-Project/
    FEATURES_DIR = PROJECT_ROOT / "results" / "features"
    LOGS_DIR = PROJECT_ROOT / "results" / "logs"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    features_path = FEATURES_DIR / "mri_features_custom.npz"
    if not features_path.exists():
        raise SystemExit(
            f"File feature non trovato: {features_path}\n"
            "Esegui prima l'estrazione delle features (Extract_Features_Custom_CNN.py)."
        )

    # ==========================
    # LOAD FEATURES
    # ==========================
    data = np.load(features_path, allow_pickle=True)
    X_train = data["train_feats"]
    y_train = data["train_labels"]
    X_val = data["val_feats"]
    y_val = data["val_labels"]
    classes = data["classes"].tolist()

    # ==========================
    # DATALOADERS
    # ==========================
    feat_train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    feat_val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    feat_train_loader = DataLoader(feat_train_data, batch_size=32, shuffle=True)
    feat_val_loader = DataLoader(feat_val_data, batch_size=32, shuffle=False)

    # ==========================
    # MODEL CONFIG
    # ==========================
    INPUT_SIZE = X_train.shape[1]
    NUM_CLASSES = len(classes)
    HIDDEN_UNITS = 64  

    # Class weights (bilanciamento classi)
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)
        
    # ==========================
    # TRAIN
    # ==========================
    mri_mlp_classifier = MLP_Softmax_Classifier(
        in_features=INPUT_SIZE,
        hidden_units=HIDDEN_UNITS,
        out_classes=NUM_CLASSES,
    ).to(device)

    mri_mlp_classifier = train_classifier(
        mri_mlp_classifier,
        train_loader=feat_train_loader,
        test_loader=feat_val_loader,
        class_weights=class_weights,
        exp_name="mri_mlp_classifier",
        logdir=str(LOGS_DIR),  # il trainer farà join(logdir, exp_name)
    )

    # ==========================
    # FINAL EVAL
    # ==========================
    mri_mlp_classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(feat_val_loader, desc="Valutazione"):
            features = features.to(device)
            outputs = mri_mlp_classifier(features)
            pred = outputs.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nAccuracy Finale: {val_accuracy * 100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=classes))
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()