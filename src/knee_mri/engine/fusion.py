"""Average within each resolution, then give both resolutions equal weight."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ..common import read_json, write_json
from .metrics import segmentation_scores, mean_scores


def align_probability(probability, target_size):
    """Validate a five-class probability map and resize it bilinearly.

    Args:
        probability: NumPy array or tensor [5,H,W], summing to one per pixel.
        target_size: Target (height, width).

    Returns:
        A float32 tensor [5,H_target,W_target] with align_corners=False.

    Raises:
        ValueError: Shape is invalid, probabilities are negative or nonfinite,
            or class sums differ from one beyond tolerance.
    """
    probability = torch.as_tensor(probability, dtype=torch.float32)
    if probability.ndim != 3 or probability.shape[0] != 5:
        raise ValueError("Expected a five-class probability tensor [5,H,W]")
    if not torch.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("Probability values must be finite and nonnegative")
    if not torch.allclose(probability.sum(0), torch.ones_like(probability[0]), atol=1e-3):
        raise ValueError("Class probabilities must sum to one")
    return F.interpolate(probability[None], size=target_size, mode="bilinear", align_corners=False)[0]


def fuse_probabilities(low, high, target_size=(512, 512)):
    """Average within resolution groups, then equally average nonempty groups.

    Args:
        low: List of low-resolution softmax maps [5,H,W].
        high: List of high-resolution maps; either group may be empty.
        target_size: Common (height, width) after interpolation.

    Returns:
        A fused probability tensor [5,H_target,W_target]. Group weights do not
        depend on the number of models in each group.

    Raises:
        ValueError: Both groups are empty or any probability map fails validation.
    """
    averages = [torch.stack([align_probability(p, target_size) for p in group]).mean(0) for group in (low, high) if group]
    if not averages:
        raise ValueError("No probabilities to fuse")
    return torch.stack(averages).mean(0)


def fuse_exports(config, low_dirs, high_dirs, output):
    """Match evaluation exports by sample identity, fuse, and recompute metrics.

    Args:
        config: Fixed-size 2.5D configuration defining labels, split, and target size.
        low_dirs: Low-resolution evaluation export directories.
        high_dirs: High-resolution evaluation export directories.
        output: Fused result directory.

    Returns:
        Aggregated segmentation metrics. Also saves probabilities, predictions,
        a manifest, and metrics.

    Raises:
        ValueError: Model type, resize mode, sample order, subset, or data paths differ.
    """
    from ..data.loaders import make_dataset
    paths = [Path(p) for p in [*low_dirs, *high_dirs]]
    if not paths:
        raise ValueError("Supply at least one probability export directory")
    manifests = [read_json(p / "manifest.json") for p in paths]
    first = manifests[0]
    ids = [r["sample"] for r in first["samples"]]
    if config["model_type"] != "2.5d" or not config.get("resize", True):
        raise ValueError("Fusion requires resized 2.5D datasets")
    for manifest in manifests:
        if manifest["model_type"] != "2.5d" or not manifest["resize"]:
            raise ValueError("Fusion requires resized 2.5D exports")
        if [r["sample"] for r in manifest["samples"]] != ids:
            raise ValueError("Export sample identity/order mismatch")
        if manifest["subset"] != first["subset"] or any(Path(manifest[k]).resolve() != Path(config[k]).resolve() for k in ("image_dir", "mask_dir")):
            raise ValueError("Exports refer to a different subset or crop/annotation directory")
    dataset = make_dataset(config, first["subset"])
    if dataset.filenames != ids:
        raise ValueError("Configured split does not match probability exports")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scores, rows = [], []
    for index, sample in enumerate(ids):
        maps = [np.load(path / manifest["samples"][index]["probability"], allow_pickle=False) for path, manifest in zip(paths, manifests)]
        fused = fuse_probabilities(maps[:len(low_dirs)], maps[len(low_dirs):], tuple(config["input_size"]))
        _, target = dataset[index]
        prediction = fused.argmax(0)
        score = segmentation_scores(prediction, target)
        scores.append(score)
        stem = Path(sample).stem
        row = {"sample": sample, "probability": f"{stem}_prob.npy", "prediction": f"{stem}_prediction.npy", **score}
        np.save(output / row["probability"], fused.numpy())
        np.save(output / row["prediction"], prediction.numpy().astype(np.uint8))
        rows.append(row)
    summary = mean_scores(scores)
    write_json(output / "metrics.json", summary)
    write_json(output / "manifest.json", {**first, "input_size": config["input_size"], "checkpoint": "probability_fusion", "samples": rows, "sources": [str(p) for p in paths]})
    return summary
