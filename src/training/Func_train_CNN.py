from os.path import join
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import torch
from torch import nn
from tqdm import tqdm

from data.Average_for_Batch_Value import AverageValueMeter


def train_classifier(
    model,
    train_loader,
    test_loader,
    class_weights,
    exp_name="experiment_cnn_classifier",
    lr=0.001,
    epochs=20,
    logdir=None,    # se None -> results/logs
    patience=4
):
    # ==========================
    # PROJECT PATHS (results/)
    # ==========================
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ML-Project/
    RESULTS_DIR = PROJECT_ROOT / "results"
    WEIGHTS_DIR = RESULTS_DIR / "weights"
    LOGS_BASE_DIR = RESULTS_DIR / "logs"

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_BASE_DIR.mkdir(parents=True, exist_ok=True)

    if logdir is None:
        logdir = str(LOGS_BASE_DIR)

    # TensorBoard logs in results/logs/<exp_name>/
    writer = SummaryWriter(join(logdir, exp_name))

    # ==========================
    # LOSS / OPTIM / SCHED
    # ==========================
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=patience
    )

    # ==========================
    # METERS
    # ==========================
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()

    # ==========================
    # DEVICE
    # ==========================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    loader = {"train": train_loader, "test": test_loader}

    global_step = 0
    best_acc = 0.0
    patience_counter = 0

    best_model_path = WEIGHTS_DIR / f"{exp_name}_BEST.pth"
    last_model_path = WEIGHTS_DIR / f"{exp_name}_LAST.pth"

    print(f"Inizio training su device: {device}")
    print(f"Log dir: {join(logdir, exp_name)}")
    print(f"Weights dir: {WEIGHTS_DIR}")

    for e in range(epochs):
        print(f"\n--- Epoch {e+1}/{epochs} ---")

        for mode in ["train", "test"]:
            loss_meter.reset()
            acc_meter.reset()
            model.train() if mode == "train" else model.eval()

            pbar = tqdm(loader[mode], desc=f"{mode.capitalize()}", unit="batch", leave=True)

            with torch.set_grad_enabled(mode == "train"):
                for x, y in pbar:
                    x = x.to(device)
                    y = y.to(device)

                    output = model(x)
                    l = criterion(output, y)

                    if mode == "train":
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += x.shape[0]

                    pred = output.argmax(dim=1)
                    acc = (pred == y).float().mean().item()

                    loss_meter.add(l.item(), x.shape[0])
                    acc_meter.add(acc, x.shape[0])

                    if mode == "train":
                        writer.add_scalar("loss/train_step", l.item(), global_step=global_step)
                        writer.add_scalar("accuracy/train_step", acc, global_step=global_step)

                    pbar.set_postfix({
                        "Loss": f"{loss_meter.value():.4f}",
                        "Acc": f"{acc_meter.value():.4f}"
                    })

            writer.add_scalar(f"loss/{mode}", loss_meter.value(), global_step=global_step)
            writer.add_scalar(f"accuracy/{mode}", acc_meter.value(), global_step=global_step)

            if mode == "test":
                val_accuracy = acc_meter.value()
                scheduler.step(val_accuracy)

                if val_accuracy > best_acc:
                    tqdm.write(
                        f" -> MIGLIORAMENTO! Acc: {best_acc:.4f} -> {val_accuracy:.4f}. Salvataggio BEST..."
                    )
                    best_acc = val_accuracy
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    tqdm.write(f" -> Nessun miglioramento. Patience: {patience_counter} / {patience}")

        torch.save(model.state_dict(), last_model_path)

        if patience_counter >= patience:
            print("\n--- EARLY STOPPING: Stop training ---")
            break

    if best_model_path.exists():
        print(f"\nFine Training. Ricarico modello migliore (Acc: {best_acc:.4f})")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    writer.close()
    return model
