"""Headless segmentation panels, entropy maps, and learning curves."""
import csv
from pathlib import Path
import numpy as np
import torch

PALETTE = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 255], [255, 165, 0], [116, 20, 12]], dtype=np.uint8)
CLASS_NAMES = ["Background", "Distal femoral cartilage", "Proximal tibial cartilage", "Patellar cartilage", "Meniscus"]


def entropy_map(probability):
    """Compute pixelwise natural-log entropy from class probabilities.

    Args:
        probability: Valid probability array or tensor [5,H,W].

    Returns:
        A NumPy entropy map [H,W] in nats. Log inputs are clamped to at least 1e-12.
    """
    p = torch.as_tensor(probability).float()
    return -(p * p.clamp_min(1e-12).log()).sum(dim=0).cpu().numpy()


def high_entropy_mask(entropy, top_percent=20):
    """Select high-entropy pixels using a percentile threshold.

    Args:
        entropy: Two-dimensional entropy map.
        top_percent: Percentage of highest-entropy pixels to select, in (0,100].

    Returns:
        A boolean mask of the same shape. Tied values at the threshold may cause
        more pixels to be selected.

    Raises:
        ValueError: The percentage is outside the valid range.
    """
    if not 0 < top_percent <= 100:
        raise ValueError("top_percent must lie in (0,100]")
    return entropy >= np.quantile(entropy, 1 - top_percent / 100)


def pyplot():
    """Enable the headless Agg backend and return the pyplot module."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_panel(image, probability, output, target=None, mask_background=False):
    """Save input, ground truth, prediction, and segmentation/entropy overlays.

    Args:
        image: Central grayscale image [H,W].
        probability: Five-class probabilities [5,H,W].
        output: Panel image path, usually ending in .png.
        target: Optional ground-truth mask [H,W] aligned with the image.
        mask_background: Restrict the entropy overlay to target or predicted foreground.

    Notes:
        Saves the panel, complete entropy NPY, and high-entropy mask NPY.

    Raises:
        ValueError: Image, probability, or target spatial shapes differ.
    """
    plt = pyplot()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    probability = np.asarray(probability)
    prediction = probability.argmax(0)
    if image.shape != prediction.shape or (target is not None and target.shape != prediction.shape):
        raise ValueError("Image/probability/target must have matching spatial shapes")
    entropy = entropy_map(probability)
    foreground = (target != 0) & (target != -100) if target is not None else prediction != 0
    masked = np.ma.masked_where(~foreground, entropy) if mask_background else entropy
    columns = 5 if target is not None else 4
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Center slice")
    index = 1
    if target is not None:
        axes[index].imshow(PALETTE[np.clip(target, 0, 4)])
        axes[index].set_title("Ground truth")
        index += 1
    axes[index].imshow(PALETTE[prediction])
    axes[index].set_title("Prediction")
    axes[index + 1].imshow(image, cmap="gray")
    axes[index + 1].imshow(np.ma.masked_where(prediction == 0, prediction), cmap=__import__("matplotlib").colors.ListedColormap(PALETTE / 255), vmin=0, vmax=4, alpha=0.4)
    axes[index + 1].set_title("Segmentation overlay")
    axes[index + 2].imshow(image, cmap="gray")
    heat = axes[index + 2].imshow(masked, cmap="jet", vmin=0, vmax=np.log(5), alpha=0.5)
    axes[index + 2].set_title("Entropy (nats)")
    fig.colorbar(heat, ax=axes[index + 2], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.savefig(output, bbox_inches="tight", dpi=150)
    plt.close(fig)
    np.save(output.with_name(output.stem + "_entropy.npy"), entropy)
    np.save(output.with_name(output.stem + "_high_entropy.npy"), high_entropy_mask(entropy))


def plot_curves(log_path, output):
    """Read per-epoch CSV logs and save training/validation learning curves.

    Args:
        log_path: CSV containing epoch and train/val loss, dice, and iou columns.
        output: Plot output path; missing parent directories are created.

    Notes:
        Saves the three-metric comparison panel and closes the figure.

    Raises:
        ValueError: The log contains no data rows.
    """
    plt = pyplot()
    with Path(log_path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Training log is empty")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, ("loss", "dice", "iou")):
        for subset in ("train", "val"):
            ax.plot([int(r["epoch"]) for r in rows], [float(r[f"{subset}_{key}"]) for r in rows], label=subset)
        ax.set(xlabel="Epoch", ylabel=key)
        ax.legend()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=150)
    plt.close(fig)


def visualize_exports(config, export_dir, output, limit=20, mask_background=False):
    """Validate an evaluation export and save a batch of 2.5D result panels.

    Args:
        config: Experiment matching the export resolution, resize mode, and data paths.
        export_dir: Directory containing manifest.json and per-sample probabilities.
        output: Plot output directory.
        limit: Maximum sample count to process, normally a positive integer.
        mask_background: Restrict entropy overlays to foreground.

    Notes:
        Saves panels and entropy arrays using sample names.

    Raises:
        ValueError: The model is 3D or configuration and sample lists do not match.
    """
    from ..data.loaders import make_dataset
    from ..common import read_json
    export_dir, output = Path(export_dir), Path(output)
    manifest = read_json(export_dir / "manifest.json")
    if config["model_type"] != "2.5d":
        raise ValueError("Export panels support 2.5D slices; use save_panel on a chosen 3D plane")
    if tuple(config["input_size"]) != tuple(manifest["input_size"]) or config.get("resize", True) != manifest["resize"]:
        raise ValueError("Visualization config must match the export resolution and resize mode")
    if any(Path(config[k]).resolve() != Path(manifest[k]).resolve() for k in ("image_dir", "mask_dir")):
        raise ValueError("Visualization crop/annotation directories differ from export")
    dataset = make_dataset(config, manifest["subset"])
    if dataset.filenames != [r["sample"] for r in manifest["samples"]]:
        raise ValueError("Visualization split differs from export")
    for index, row in enumerate(manifest["samples"][:limit]):
        image, target = dataset[index]
        probability = np.load(export_dir / row["probability"], allow_pickle=False)
        save_panel(image[1].numpy(), probability, output / (Path(row["sample"]).stem + ".png"), target.numpy(), mask_background)
