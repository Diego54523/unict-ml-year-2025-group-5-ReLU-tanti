import os
import random
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms


class RadNetRunner:
    # ==========================
    # CONFIG (default identici)
    # ==========================
    SEED = 42

    BATCH_SIZE = 64
    NUM_EPOCHS = 40
    EARLY_STOP_PATIENCE = 7

    LR_HEAD = 5e-5
    LR_LAYER4 = 1e-5
    LR_LAYER3 = 5e-6

    WEIGHT_DECAY = 1e-4

    K_FOLDS = 5                 # >= 5
    INNER_VAL_SPLIT = 0.15      # validation interna (dal train del fold)

    NUM_WORKERS = 8
    USE_AMP = True

    FREEZE_ALL_BUT_LAYER4 = True
    UNFREEZE_LAYER3 = True

    LABEL_SMOOTHING = 0.05
    GRAD_CLIP_NORM = 1.0

    # ==========================
    # DATASET PATH (robusto)
    # ==========================  
    load_dotenv()
    
    DATA_DIR = os.getenv("DATA_PATH")
    if not DATA_DIR:
        raise ValueError("DATA_PATH non definito nel file .env")  
    

    def __init__(
        self,
        data_dir: str | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
        num_epochs: int | None = None,
        k_folds: int | None = None,
        num_workers: int | None = None,
        use_amp: bool | None = None,
    ):
        # Override opzionali (senza cambiare logica)
        if data_dir is not None:
            self.DATA_DIR = data_dir
        if seed is not None:
            self.SEED = seed
        if batch_size is not None:
            self.BATCH_SIZE = batch_size
        if num_epochs is not None:
            self.NUM_EPOCHS = num_epochs
        if k_folds is not None:
            self.K_FOLDS = k_folds
        if num_workers is not None:
            self.NUM_WORKERS = num_workers
        if use_amp is not None:
            self.USE_AMP = use_amp

        self.seed_everything(self.SEED)
        self.SCALER_CTOR, self.AUTOCAST_CTX = self.make_amp()

        # Transforms (MRI-friendly) identici
        self.train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05),
                    shear=5
                )
            ], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.val_test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ==========================
    # SEED
    # ==========================
    @staticmethod
    def seed_everything(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # ==========================
    # AMP (API nuova con fallback)
    # ==========================
    @staticmethod
    def make_amp():
        from torch.amp import GradScaler, autocast
        try:
            def autocast_ctx(enabled: bool):
                return autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)

            def scaler_ctor(enabled: bool):
                return GradScaler("cuda", enabled=enabled)

            return scaler_ctor, autocast_ctx

        except Exception:
            def autocast_ctx(enabled: bool):
                return autocast(enabled=enabled)

            def scaler_ctor(enabled: bool):
                return GradScaler(enabled=enabled)

            return scaler_ctor, autocast_ctx

    # ==========================
    # DATASET: subset con transform indipendente
    # ==========================
    class SubsetWithTransform(Dataset):
        def __init__(self, base_dataset, indices, transform=None):
            self.base_dataset = base_dataset
            self.indices = list(indices)
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            x, y = self.base_dataset[self.indices[idx]]  # PIL.Image
            if self.transform is not None:
                x = self.transform(x)  # Tensor
            return x, y

    # ==========================
    # STRATIFIED SPLIT (senza sklearn) su targets "flat"
    # ==========================
    @staticmethod
    def stratified_split_indices(targets, val_split=0.15, test_split=0.15, seed=42):
        """
        targets: array-like di label (0..C-1) della "popolazione" da splittare.
        Ritorna indici RELATIVI [0..len(targets)-1] per train/val/test.
        """
        targets = np.asarray(targets)
        classes = np.unique(targets)

        rng = np.random.default_rng(seed)
        train_idx, val_idx, test_idx = [], [], []

        for c in classes:
            idx_c = np.where(targets == c)[0]
            rng.shuffle(idx_c)

            n_c = len(idx_c)
            n_test = int(round(test_split * n_c))
            n_val = int(round(val_split * n_c))

            # clamp se per arrotondamenti sfori
            if n_test + n_val > n_c:
                n_val = max(n_c - n_test, 0)
                if n_test + n_val > n_c:
                    n_test = max(n_c - n_val, 0)

            n_train = n_c - n_val - n_test

            # se resta 0 train e ci sono campioni, sposta da val o test
            if n_c > 0 and n_train <= 0:
                if n_val > 0:
                    n_val -= 1
                    n_train += 1
                elif n_test > 0:
                    n_test -= 1
                    n_train += 1

            train_idx.extend(idx_c[:n_train].tolist())
            val_idx.extend(idx_c[n_train:n_train + n_val].tolist())
            test_idx.extend(idx_c[n_train + n_val:n_train + n_val + n_test].tolist())

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)
        return train_idx, val_idx, test_idx

    @classmethod
    def stratified_split_from_indices(cls, indices, full_targets, val_split=0.15, seed=42):
        """
        indices: indici ORIGINALI del dataset base (es: outer-train del fold)
        full_targets: base_dataset.targets
        ritorna train_idx/val_idx ORIGINALI
        """
        indices = np.asarray(indices, dtype=np.int64)
        t = np.asarray(full_targets)[indices]
        train_rel, val_rel, _ = cls.stratified_split_indices(
            t, val_split=val_split, test_split=0.0, seed=seed
        )
        train_idx = indices[train_rel].tolist()
        val_idx = indices[val_rel].tolist()
        return train_idx, val_idx

    # ==========================
    # STRATIFIED K-FOLD (senza sklearn)
    # ==========================
    @staticmethod
    def stratified_kfold_indices(targets, k=5, seed=42, shuffle=True):
        """
        targets: labels dell'intero dataset
        ritorna: lista di k liste di indici ORIGINALI (folds)
        """
        targets = np.asarray(targets)
        classes = np.unique(targets)
        rng = np.random.default_rng(seed)

        folds = [[] for _ in range(k)]

        for c in classes:
            idx_c = np.where(targets == c)[0]
            if shuffle:
                rng.shuffle(idx_c)
            for i, idx in enumerate(idx_c):
                folds[i % k].append(int(idx))

        if shuffle:
            for f in folds:
                rng.shuffle(f)

        return folds

    # ==========================
    # MODEL
    # ==========================
    def get_model(self, num_classes: int):
        backbone = torch.hub.load(
            "Warvito/radimagenet-models",
            model="radimagenet_resnet50",
            verbose=True,
            trust_repo=True,
        )

        if hasattr(backbone, "fc"):
            backbone.fc = nn.Identity()

        class RadResNet50Classifier(nn.Module):
            def __init__(self, backbone, num_classes):
                super().__init__()
                self.backbone = backbone
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.dropout = nn.Dropout(0.5)
                self.fc = nn.Linear(2048, num_classes)

            def forward(self, x):
                x = self.backbone(x)
                if x.dim() == 4:
                    x = self.pool(x)
                    x = torch.flatten(x, 1)
                x = self.dropout(x)
                x = self.fc(x)
                return x

        model = RadResNet50Classifier(backbone, num_classes)

        if self.FREEZE_ALL_BUT_LAYER4:
            for p in model.backbone.parameters():
                p.requires_grad = False

            if hasattr(model.backbone, "layer4"):
                for p in model.backbone.layer4.parameters():
                    p.requires_grad = True

            if self.UNFREEZE_LAYER3 and hasattr(model.backbone, "layer3"):
                for p in model.backbone.layer3.parameters():
                    p.requires_grad = True

        return model

    # ==========================
    # EVAL + METRICHE
    # ==========================
    @torch.no_grad()
    def evaluate(self, model, loader, criterion, device, n_classes):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        per_class_correct = np.zeros(n_classes, dtype=np.int64)
        per_class_total = np.zeros(n_classes, dtype=np.int64)

        for inputs, labels in tqdm(loader, desc="Eval", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for c in range(n_classes):
                mask = (labels == c)
                per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                per_class_total[c] += mask.sum().item()

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)

        recalls = []
        for c in range(n_classes):
            if per_class_total[c] > 0:
                recalls.append(per_class_correct[c] / per_class_total[c])
        balanced_acc = float(np.mean(recalls)) if len(recalls) else 0.0

        return epoch_loss, epoch_acc, balanced_acc, per_class_correct, per_class_total

    # ==========================
    # TRAIN (AMP + BN freeze + grad clip)
    # ==========================
    def train_one_epoch(self, model, loader, criterion, optimizer, device, scaler, use_amp: bool):
        model.train()

        # Blocca running stats delle BatchNorm nel backbone congelato:
        if hasattr(model, "backbone") and self.FREEZE_ALL_BUT_LAYER4:
            model.backbone.eval()
            if hasattr(model.backbone, "layer4"):
                model.backbone.layer4.train()
            if self.UNFREEZE_LAYER3 and hasattr(model.backbone, "layer3"):
                model.backbone.layer3.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(loader, desc="Train", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with self.AUTOCAST_CTX(enabled=True):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.GRAD_CLIP_NORM)
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    # ==========================
    # RUN: 5-FOLD STRATIFIED CV (main originale)
    # ==========================
    def run(self):
        assert os.path.isdir(self.DATA_DIR), f"DATA_DIR non trovato: {self.DATA_DIR}"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)

        base_dataset = datasets.ImageFolder(root=self.DATA_DIR, transform=None)
        class_names = base_dataset.classes
        num_classes = len(class_names)
        targets = base_dataset.targets

        print("Classi trovate:", class_names, "| num_classes:", num_classes)

        K = max(5, int(self.K_FOLDS))
        folds = self.stratified_kfold_indices(targets, k=K, seed=self.SEED, shuffle=True)

        fold_metrics = []  # (test_loss, test_acc, test_balacc)

        common_loader_kwargs = dict(
            num_workers=self.NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(self.NUM_WORKERS > 0),
            prefetch_factor=2 if self.NUM_WORKERS > 0 else None
        )

        def dl_kwargs():
            return {k: v for k, v in common_loader_kwargs.items() if v is not None}

        for fold in range(K):
            print("\n" + "=" * 34)
            print(f"           FOLD {fold + 1}/{K}")
            print("=" * 34)

            outer_test_idx = folds[fold]
            outer_train_idx = [i for f in range(K) if f != fold for i in folds[f]]

            # Validation interna (per early stopping) estratta SOLO dall'outer-train
            train_idx, val_idx = self.stratified_split_from_indices(
                outer_train_idx, targets, val_split=self.INNER_VAL_SPLIT, seed=self.SEED + fold
            )

            train_dataset = self.SubsetWithTransform(base_dataset, train_idx, transform=self.train_transform)
            val_dataset = self.SubsetWithTransform(base_dataset, val_idx, transform=self.val_test_transform)
            test_dataset = self.SubsetWithTransform(base_dataset, outer_test_idx, transform=self.val_test_transform)

            train_targets = np.array([targets[i] for i in train_idx], dtype=np.int64)
            class_counts = np.bincount(train_targets, minlength=num_classes).astype(np.float32)
            print("Campioni TRAIN (outer) per classe:", class_counts)

            # Pesi SOLO per il sampler (non per la loss)
            class_weights = class_counts.sum() / (num_classes * np.clip(class_counts, 1.0, None))
            sample_weights = class_weights[train_targets]

            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.BATCH_SIZE,
                sampler=sampler,
                shuffle=False,
                **dl_kwargs()
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                **dl_kwargs()
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                **dl_kwargs()
            )

            # Model nuovo ad ogni fold
            model = self.get_model(num_classes).to(device)

            # Loss non pesata + label smoothing
            criterion = nn.CrossEntropyLoss(label_smoothing=self.LABEL_SMOOTHING)

            # Optimizer con LR differenziati
            param_groups = [{"params": model.fc.parameters(), "lr": self.LR_HEAD}]

            if hasattr(model.backbone, "layer4"):
                layer4_params = [p for p in model.backbone.layer4.parameters() if p.requires_grad]
                if layer4_params:
                    param_groups.append({"params": layer4_params, "lr": self.LR_LAYER4})

            if self.UNFREEZE_LAYER3 and hasattr(model.backbone, "layer3"):
                layer3_params = [p for p in model.backbone.layer3.parameters() if p.requires_grad]
                if layer3_params:
                    param_groups.append({"params": layer3_params, "lr": self.LR_LAYER3})

            optimizer = optim.AdamW(param_groups, weight_decay=self.WEIGHT_DECAY)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2
            )

            use_amp = self.USE_AMP and (device.type == "cuda")
            scaler = self.SCALER_CTOR(enabled=use_amp)

            best_val_balacc = 0.0
            epochs_no_improve = 0
            best_path = f"best_radimagenet_resnet50_fold{fold + 1}.pth"

            for epoch in range(1, self.NUM_EPOCHS + 1):
                print(f"\n===== FOLD {fold + 1} | EPOCH {epoch}/{self.NUM_EPOCHS} =====")
                print("LRs:", [pg["lr"] for pg in optimizer.param_groups])

                train_loss, train_acc = self.train_one_epoch(
                    model, train_loader, criterion, optimizer, device, scaler, use_amp
                )
                val_loss, val_acc, val_balacc, _, _ = self.evaluate(
                    model, val_loader, criterion, device, num_classes
                )

                print(f"Train - loss: {train_loss:.4f} | acc: {train_acc:.4f}")
                print(f"Val   - loss: {val_loss:.4f} | acc: {val_acc:.4f} | bal_acc: {val_balacc:.4f}")

                scheduler.step(val_balacc)

                if val_balacc > best_val_balacc:
                    best_val_balacc = val_balacc
                    torch.save(model.state_dict(), best_path)
                    print(f"--> Nuovo best fold salvato: {best_path}")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"Nessun miglioramento per {epochs_no_improve} epoche.")

                if epochs_no_improve >= self.EARLY_STOP_PATIENCE:
                    print("Early stopping: val balanced accuracy non migliora.")
                    break

            # TEST = fold esterno
            print("\nCarico il best model del fold per il test esterno...")
            model.load_state_dict(torch.load(best_path, map_location=device))

            test_loss, test_acc, test_balacc, _, _ = self.evaluate(
                model, test_loader, criterion, device, num_classes
            )

            print(f"\n===== FOLD {fold + 1} TEST (outer) =====")
            print(f"Loss: {test_loss:.4f}")
            print(f"Accuracy: {test_acc:.4f}")
            print(f"Balanced Accuracy: {test_balacc:.4f}")

            fold_metrics.append((test_loss, test_acc, test_balacc))

        # Riassunto CV
        losses = np.array([m[0] for m in fold_metrics], dtype=np.float64)
        accs = np.array([m[1] for m in fold_metrics], dtype=np.float64)
        bals = np.array([m[2] for m in fold_metrics], dtype=np.float64)

        print("\n" + "=" * 34)
        print("      RISULTATI K-FOLD CV")
        print("=" * 34)
        print(f"K = {K}")
        print(f"Test Loss          : {losses.mean():.4f} ± {losses.std(ddof=1):.4f}")
        print(f"Test Accuracy      : {accs.mean():.4f} ± {accs.std(ddof=1):.4f}")
        print(f"Test Balanced Acc  : {bals.mean():.4f} ± {bals.std(ddof=1):.4f}")


if __name__ == "__main__":
    RadNetRunner().run()