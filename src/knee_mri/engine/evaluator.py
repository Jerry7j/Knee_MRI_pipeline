"""Evaluate best weights separately; preserve sample identity in probability exports."""
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from ..data.loaders import make_loader
from ..models.factory import build_model, load_weights
from ..common import get_device
from ..common import write_json
from .metrics import segmentation_scores, mean_scores


def evaluate(config, checkpoint=None, output=None, subset="test", save_probs=True):
    """Evaluate a fixed subset independently and export results by sample name.

    Args:
        config: Segmentation experiment configuration with resolved paths.
        checkpoint: Optional weights; defaults to the configured or experiment-best model.
        output: Optional export directory; defaults to the experiment subset directory.
        subset: val or test.
        save_probs: Save softmax probabilities for later fusion and entropy plots.

    Returns:
        Aggregated Dice/IoU, per-class metrics, and sample count.

    Notes:
        Saves prediction arrays, optional probability arrays, manifest.json,
        and metrics.json. Removes bottom/right batch padding for 2.5D exports;
        3D exports retain the configured volume size.
    """
    device = get_device(config.get("device", "auto"))
    checkpoint = checkpoint or config.get("checkpoint") or str(Path(config["output_dir"]) / "best_model.pth")
    model = build_model(config).to(device)
    load_weights(model, checkpoint, device)
    model.eval()
    loader = make_loader(config, subset)
    output = Path(output or Path(config["output_dir"]) / subset)
    output.mkdir(parents=True, exist_ok=True)
    rows, scores = [], []
    cursor = 0
    with torch.inference_mode():
        for x, y in tqdm(loader, desc=subset):
            probabilities = model(x.to(device)).softmax(1).cpu()
            for probability, target in zip(probabilities, y):
                sample = loader.dataset.filenames[cursor]
                cursor += 1
                # Remove right/bottom batch padding from baseline exports.
                if config["model_type"] == "2.5d":
                    valid = target != -100
                    height = int(valid.any(1).sum())
                    width = int(valid.any(0).sum())
                    probability, target = probability[:, :height, :width], target[:height, :width]
                prediction = probability.argmax(0)
                score = segmentation_scores(prediction, target)
                scores.append(score)
                stem = Path(sample).stem
                np.save(output / f"{stem}_prediction.npy", prediction.numpy().astype(np.uint8))
                row = {"sample": sample, "prediction": f"{stem}_prediction.npy", **score}
                if save_probs:
                    row["probability"] = f"{stem}_prob.npy"
                    np.save(output / row["probability"], probability.numpy())
                rows.append(row)
    summary = mean_scores(scores)
    write_json(output / "metrics.json", summary)
    write_json(output / "manifest.json", {"model_type": config["model_type"], "subset": subset,
               "image_dir": config["image_dir"], "mask_dir": config["mask_dir"], "resize": config.get("resize", True),
               "input_size": config["input_size"], "checkpoint": str(checkpoint), "samples": rows})
    return summary
