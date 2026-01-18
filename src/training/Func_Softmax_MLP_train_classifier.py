from os.path import join
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
import torch
from sklearn.metrics import accuracy_score
from data.Average_for_Batch_Value import AverageValueMeter
import os

def train_classifier(model, train_loader, test_loader, class_weights, exp_name = "experiment_mlp_classifier", lr = 0.001, epochs = 100, momentum = 0.9, logdir = "logs_mlp_classifier"):
    if class_weights is not None:
        criterion = nn.NLLLoss(weight = class_weights) #Usiamo NLLLoss(Negative Log Likelihood Loss) perché il modello usa LogSoftmax nell'ultimo layer
        # Questa funzione di Loss banalmente, prende un vettore di output della rete (log-probabilità) e le etichette vere, e calcola: Loss = -log(P_classe_corretta),
        # ma essendo che la nostra rete calcola già il logaritmo, essendo LogSoftmax, la formula si riduce a Loss = -x_y,
        # ossia il negativo, che lavorando col numeri già negativi a questo punto saranno positivi, della predizione del modello per la classe giusta del sample indagato.
    else:
        criterion = nn.NLLLoss()

    optimizer = SGD(model.parameters(), lr, momentum=momentum)
   
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()

    writer = SummaryWriter(join(logdir, exp_name))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    weights_dir = os.path.join(script_dir, "weights_Softmax_MLP")

    os.makedirs(weights_dir, exist_ok = True)

    # definiamo un dizionario contenente i loader di training e test
    loader = {"train": train_loader, "test": test_loader}

    
    global_step = 0
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

                    x = x.view(x.size(0), -1)

                    output = model(x)

                    # aggiorniamo il global_step
                    # conterrà il numero di campioni visti durante il training
                    n = x.shape[0]  # numero di elementi nel batch
                    global_step += n
                    
                    l = criterion(output, y)

                    if mode == "train":
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    acc = accuracy_score(y.to("cpu"), output.to("cpu").max(1)[1])
                    loss_meter.add(l.item(), n)
                    acc_meter.add(acc, n)

                    # loggiamo i risultati iterazione per iterazione solo durante il training
                    if mode == "train":
                        writer.add_scalar("loss/train", loss_meter.value(), global_step = global_step)
                        writer.add_scalar("accuracy/train", acc_meter.value(), global_step = global_step)
                        
            # una volta finita l'epoca (sia nel caso di training che test,
            # loggiamo le stime finali)
            writer.add_scalar("loss/" + mode, loss_meter.value(), global_step = global_step)
            writer.add_scalar("accuracy/" + mode, acc_meter.value(), global_step = global_step)

        # conserviamo i pesi del modello alla fine di un ciclo di training e test
        weigths_path = f"{exp_name}_weights_Softmax.pth"
        
        output_path = os.path.join(weights_dir, weigths_path)

        torch.save(model.state_dict(), output_path)
    return model