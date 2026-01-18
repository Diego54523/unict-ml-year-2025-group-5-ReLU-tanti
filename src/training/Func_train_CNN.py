from os.path import join
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import torch
from sklearn.metrics import accuracy_score
from data.Average_for_Batch_Value import AverageValueMeter
import os
import sys
from tqdm import tqdm 

def train_classifier(model, train_loader, test_loader, class_weights, exp_name = "experiment_mlp_classifier", lr=0.001, epochs=20, momentum=0.9, logdir="logs_mlp_classifier", patience=4):
    
    criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)

    # Setup Optimizer (Adam per convergenza veloce)
    optimizer = Adam(model.parameters(), lr = lr, weight_decay = 1e-4)

    # Setup Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = patience)
   
    # Meters
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()

    # Tensorboard & Paths
    writer = SummaryWriter(join(logdir, exp_name))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, "weights_Custom_CNN")
    os.makedirs(weights_dir, exist_ok = True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    loader = {"train": train_loader, "test": test_loader}
    
    global_step = 0
    best_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(weights_dir, f"{exp_name}_BEST.pth")

    print(f"Inizio training su device: {device}")

    for e in range(epochs):
        print(f"\n--- Epoch {e+1}/{epochs} ---")

        for mode in ["train", "test"]:
            loss_meter.reset()
            acc_meter.reset()
            model.train() if mode == "train" else model.eval()

            # Inizializziamo la barra TQDM
            # desc: Testo a sinistra della barra
            # unit: unità di misura (batch)
            # leave=True: lascia la barra stampata alla fine
            pbar = tqdm(loader[mode], desc = f"{mode.capitalize()}", unit = "batch", leave = True)
            
            with torch.set_grad_enabled(mode == "train"):
                for batch in pbar:
                    x = batch[0].to(device)
                    y = batch[1].to(device)

                    output = model(x)
                    l = criterion(output, y)

                    if mode == "train":
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        global_step += x.shape[0]

                    # Calcolo metriche
                    acc = accuracy_score(y.cpu(), output.cpu().max(1)[1])
                    loss_meter.add(l.item(), x.shape[0])
                    acc_meter.add(acc, x.shape[0])

                    # Logging Tensorboard (solo train step)
                    if mode == "train":
                        writer.add_scalar("loss/train_step", l.item(), global_step = global_step)
                        writer.add_scalar("accuracy/train_step", acc, global_step = global_step)

                    # Aggiorniamo la barra con le metriche correnti (media accumulata)
                    pbar.set_postfix({
                        "Loss": f"{loss_meter.value():.4f}", 
                        "Acc": f"{acc_meter.value():.4f}"
                    })
            
            # Fine del loop per questa mode
            # Scriviamo su Tensorboard i valori medi dell'epoca
            writer.add_scalar(f"loss/{mode}", loss_meter.value(), global_step = global_step)
            writer.add_scalar(f"accuracy/{mode}", acc_meter.value(), global_step = global_step)

            # Logica di salvataggio e Early Stopping (solo su test)
            if mode == "test":
                val_accuracy = acc_meter.value()
                scheduler.step(val_accuracy)

                if val_accuracy > best_acc:
                    # Usiamo tqdm.write per non rompere la barra grafica
                    tqdm.write(f" -> MIGLIORAMENTO! Acc: {best_acc:.4f} -> {val_accuracy:.4f}. Salvataggio...")
                    best_acc = val_accuracy
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    tqdm.write(f" -> Nessun miglioramento. Patience: {patience_counter} / {patience}")

        # Controllo Early Stopping
        if patience_counter >= patience:
            print(f"\n--- EARLY STOPPING: Stop training ---")
            break

    # Caricamento finale
    if os.path.exists(best_model_path):
        print(f"\nFine Training. Ricarico modello migliore (Acc: {best_acc:.4f})")
        model.load_state_dict(torch.load(best_model_path))
    
    return model