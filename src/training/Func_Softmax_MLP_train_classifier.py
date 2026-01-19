from os.path import join
from pathlib import Path
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
import torch
from data.Average_for_Batch_Value import AverageValueMeter


def train_classifier(
    model,
    train_loader,
    test_loader,
    class_weights,
    exp_name="experiment_mlp_classifier",
    lr=0.001,
    epochs=100,
    momentum=0.9,
    logdir=None,  # se None, usa results/logs
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

    # TensorBoard log: results/logs/<exp_name>/
    writer = SummaryWriter(join(logdir, exp_name))

    # ==========================
    # LOSS (coerente con LogSoftmax)
    # ==========================
    if class_weights is not None:
        criterion = nn.NLLLoss(weight=class_weights)
    else:
        criterion = nn.NLLLoss()

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    loader = {"train": train_loader, "test": test_loader}

    global_step = 0

    best_val_loss = float("inf")
    best_path = WEIGHTS_DIR / f"{exp_name}_BEST.pth"
    last_path = WEIGHTS_DIR / f"{exp_name}_LAST.pth"

    for e in range(epochs):
        print(f"Epoch {e+1} of {epochs}")

        epoch_val_loss = None

        for mode in ["train", "test"]:
            loss_meter.reset()
            acc_meter.reset()
            model.train() if mode == "train" else model.eval()

            with torch.set_grad_enabled(mode == "train"):
                for batch in loader[mode]:
                    x = batch[0].to(device)
                    y = batch[1].to(device)

                    x = x.view(x.size(0), -1)

                    output = model(x)

                    n = x.shape[0]
                    global_step += n

                    l = criterion(output, y)

                    if mode == "train":
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    pred = output.argmax(dim=1)
                    acc = (pred == y).float().mean().item()

                    loss_meter.add(l.item(), n)
                    acc_meter.add(acc, n)

                    if mode == "train":
                        writer.add_scalar("loss/train_iter", loss_meter.value(), global_step=global_step)
                        writer.add_scalar("accuracy/train_iter", acc_meter.value(), global_step=global_step)

            # log fine-epoca
            writer.add_scalar("loss/" + mode, loss_meter.value(), global_step=global_step)
            writer.add_scalar("accuracy/" + mode, acc_meter.value(), global_step=global_step)

            if mode == "test":
                epoch_val_loss = loss_meter.value()

        # Salva sempre ultimo
        torch.save(model.state_dict(), last_path)

        if epoch_val_loss is not None and epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_path)
            print(f"New best saved: {best_path} (val loss={best_val_loss:.6f})")

    writer.close()
    return model
