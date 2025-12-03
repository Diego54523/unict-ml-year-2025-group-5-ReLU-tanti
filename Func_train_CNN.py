from os.path import join
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
import torch
from sklearn.metrics import accuracy_score
from Average_for_Batch_Value import AverageValueMeter
import os

def train_classifier(model, train_loader, test_loader, class_weights, exp_name="experiment_mlp_classifier", lr=0.001, epochs = 10, momentum = 0.9, logdir="logs_mlp_classifier"):
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4) #Usiamo weight_decay per la regolarizzazione della CNN
   
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
            if mode == "test" and acc_meter.value() > best_acc:
                best_acc = acc_meter.value()
                best_model_path = os.path.join(weights_dir, f"{exp_name}_BEST.pth")
                torch.save(model.state_dict(), best_model_path)

        
        output_path = os.path.join(weights_dir, f"{exp_name}_Last.pth")

        torch.save(model.state_dict(), output_path)
    return model