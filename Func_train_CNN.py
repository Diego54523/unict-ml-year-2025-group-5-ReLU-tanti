from os.path import join
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
import torch
from sklearn.metrics import accuracy_score
from Average_for_Batch_Value import AverageValueMeter
import os

def train_classifier(model, train_loader, test_loader, class_weights, exp_name="experiment_mlp_classifier", lr=0.0001, epochs = 10, momentum = 0.9, logdir="logs_mlp_classifier", patience = 4):
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-3) #Usiamo weight_decay per la regolarizzazione della CNN

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=patience, verbose=True)
   
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()

    writer = SummaryWriter(join(logdir, exp_name))
    script_dir = os.path.dirname(os.path.abspath(__file__))

    weights_dir = os.path.join(script_dir, "weights")

    os.makedirs(weights_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # definiamo un dizionario contenente i loader di training e test
    loader = {"train": train_loader, "test": test_loader}

    
    global_step = 0
    best_acc = 0.0

    patience_counter = 0
    best_model_path = os.path.join(weights_dir, f"{exp_name}_BEST.pth")

    for e in range(epochs):
        print(f"Epoch {e+1} of {epochs}")

        # iteriamo tra due modalità: train e test
        for mode in ["train", "test"]:
            loss_meter.reset()
            acc_meter.reset()
            model.train() if mode == "train" else model.eval()

            # abilitiamo i gradienti solo in training
            with torch.set_grad_enabled(mode == "train"):
                for i, batch in enumerate(loader[mode]):
                    x = batch[0].to(device)  # "portiamoli sul device corretto"
                    y = batch[1].to(device)

                    output = model(x)

                    l = criterion(output, y)

                    if mode == "train":
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        n = x.shape[0]  # numero di elementi nel batch
                        global_step += n

                    acc = accuracy_score(y.cpu(), output.cpu().max(1)[1]) # sklearn lavora su CPU, quindi spostiamo le variabili di accuracy_score su CPU
                    loss_meter.add(l.item(), n)
                    acc_meter.add(acc, n)

                    # loggiamo i risultati iterazione per iterazione solo durante il training
                    if mode == "train":
                        writer.add_scalar("loss/train", loss_meter.value(), global_step=global_step)
                        writer.add_scalar("accuracy/train", acc_meter.value(), global_step=global_step)
                        
            # una volta finita l'epoca (sia nel caso di training che test,
            # loggiamo le stime finali)
            writer.add_scalar("loss/" + mode, loss_meter.value(), global_step=global_step)
            writer.add_scalar("accuracy/" + mode, acc_meter.value(), global_step=global_step)

            print(f"{mode.capitalize()} Loss: {loss_meter.value():.4f} | {mode.capitalize()} Accuracy: {acc_meter.value():.4f}")

            # conserviamo i pesi del modello alla fine
            if mode == "test":
                val_accuracy = acc_meter.value()
                val_loss = loss_meter.value()

                scheduler.step(val_accuracy)

                if val_accuracy > best_acc:
                    print(f"    [MIGLIORAMENTO] Accuracy salita da {best_acc:.4f} a {val_accuracy:.4f}. Salvataggio modello...")
                    best_acc = val_accuracy
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    print(f"    [NESSUN MIGLIORAMENTO] Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"\n--- EARLY STOPPING ATTIVATO ---")
                print(f"Il modello non migliora da {patience} epoche. Stop.")
                break

        
        print(f"\nRicaricamento dei pesi migliori (Acc: {best_acc:.4f})...")
        model.load_state_dict(torch.load(best_model_path))
    return model