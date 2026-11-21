"""Cross-entropy + Adam, best validation IoU, CSV log, and checkpoints."""
import csv
from pathlib import Path
import torch
from tqdm import tqdm
from ..data.loaders import make_loader
from .metrics import segmentation_scores, mean_scores
from ..models.factory import build_model
from ..common import seed_everything, get_device
from ..common import write_json


def run_epoch(model, loader, device, optimizer=None):
    """Run one training or validation epoch and aggregate loss and metrics.

    Args:
        model: Segmentation model producing five-class logits.
        loader: DataLoader yielding images and integer labels.
        device: PyTorch compute device.
        optimizer: Optimizer, or None to disable gradients and use validation mode.

    Returns:
        A dictionary containing loss, sample count, Dice/IoU, and per-class metrics.

    Raises:
        RuntimeError: The loss is nonfinite.
        ValueError: No valid samples are available for metric aggregation.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, total_samples, scores = 0.0, 0, []
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc="train" if training else "val"):
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; inspect input intensities and labels")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.shape[0]
            total_samples += x.shape[0]
            scores.extend(segmentation_scores(p, t) for p, t in zip(logits.detach().argmax(1), y))
    return {"loss": total_loss / max(total_samples, 1), **mean_scores(scores)}


def train(config):
    """Train a segmentation model and save the best validation-IoU checkpoint.

    Args:
        config: Resolved configuration specifying data, model, device, epochs,
            and output directory.

    Returns:
        A summary of epoch count, best validation IoU, and best weight path.

    Notes:
        Saves configuration, CSV logs, best_model.pth, best/last.ckpt, and a summary.
        Existing outputs are overwritten. Does not run independent testing or resume.
    """
    seed_everything(config.get("seed", 65))
    device = get_device(config.get("device", "auto"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    loaders = {subset: make_loader(config, subset, training=subset == "train") for subset in ("train", "val")}
    model = build_model(config, pretrained=config.get("pretrained", False)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-4))
    best_iou = -1.0
    log_path = output / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["epoch", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config.get("epochs", 30) + 1):
            results = {"train": run_epoch(model, loaders["train"], device, optimizer), "val": run_epoch(model, loaders["val"], device)}
            row = {"epoch": epoch, **{f"{subset}_{key}": result[key] for subset, result in results.items() for key in ("loss", "dice", "iou")}}
            writer.writerow(row)
            handle.flush()
            improved = results["val"]["iou"] > best_iou
            if improved:
                best_iou = results["val"]["iou"]
                torch.save(model.state_dict(), output / "best_model.pth")
            checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_iou": best_iou, "config": config}
            torch.save(checkpoint, output / "last.ckpt")
            if improved:
                torch.save(checkpoint, output / "best.ckpt")
            print(row)
    summary = {"epochs": config.get("epochs", 30), "best_val_iou": best_iou, "checkpoint": str(output / "best_model.pth")}
    write_json(output / "summary.json", summary)
    return summary
