"""Reusable 2.5D multi-resolution predictor for CLI and Streamlit."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ..models.factory import build_model, load_weights
from ..data.crops import best_bbox, normalize_uint8
from .fusion import fuse_probabilities
from ..common import read_json, require_artifact, write_json
from ..common import get_device
from ..data.inputs import load_volume, validate_volume


class EnsemblePredictor:
    def __init__(self, config_path, device="auto"):
        """Load configured segmentation models and an optional YOLO localizer.

        Args:
            config_path: Ensemble JSON containing models, target_size, and localizer.
            device: auto, cpu, or a CUDA device name.

        Notes:
            Stores evaluation-mode models, resolution groups, and the optional detector.
            Configuration, dependency, artifact, and weight compatibility errors propagate.
        """
        path = Path(config_path).resolve()
        config = read_json(path)
        root = (path.parent / config.get("project_root", "..")).resolve()
        self.device = get_device(device)
        self.target_size = tuple(config.get("target_size", [512, 512]))
        self.models = []
        for spec in config["models"]:
            size = int(spec["size"])
            if size <= 0 or size % 32 or spec["group"] not in ("low", "high"):
                raise ValueError("Each model needs size divisible by 32 and group low/high")
            checkpoint = require_artifact(root / spec["checkpoint"])
            model_config = {"model_type": "2.5d", "architecture": spec.get("architecture", "unet"),
                            "pfdnet_channels": spec.get("pfdnet_channels", 128)}
            model = build_model(model_config).to(self.device)
            load_weights(model, checkpoint, self.device)
            self.models.append((model.eval(), size, spec["group"]))
        if not self.models:
            raise ValueError("Configure at least one segmentation model")
        self.detector = None
        if config.get("localizer"):
            from ultralytics import YOLO
            self.detector = YOLO(str(require_artifact(root / config["localizer"])))

    @torch.inference_mode()
    def predict_slice(self, volume, index):
        """Segment a central slice using a three-slice multi-model ensemble.

        Args:
            volume: Finite volume array [N,H,W].
            index: Central slice index. Missing neighbors at volume edges repeat
                the central slice.

        Returns:
            A dictionary with probability [5,H_target,W_target],
            prediction [H_target,W_target], image [H_target,W_target], and bbox_xyxy
            in image coordinates. All neighbors share the central bounding box;
            without a detector, the entire image is used.

        Raises:
            IndexError: The central index is outside the volume.
            ValueError: The volume is invalid or no valid central ROI is detected.
        """
        volume = validate_volume(volume)
        if not 0 <= index < len(volume):
            raise IndexError(index)
        box = best_bbox(volume[index], self.detector) if self.detector is not None else (0, 0, volume.shape[2], volume.shape[1])
        if box is None:
            raise ValueError(f"No knee ROI detected on central slice {index}")
        # Apply the central slice's box to all three slices, preserving spatial context.
        x1, y1, x2, y2 = box
        crops = [normalize_uint8(volume[min(max(index + offset, 0), len(volume) - 1), y1:y2, x1:x2]) for offset in (-1, 0, 1)]
        groups = {"low": [], "high": []}
        inputs = {}
        for model, size, group in self.models:
            if size not in inputs:
                arrays = [np.asarray(Image.fromarray(c).resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255 for c in crops]
                inputs[size] = torch.from_numpy(np.stack(arrays))[None].to(self.device)
            groups[group].append(model(inputs[size]).softmax(1)[0].cpu())
        probability = fuse_probabilities(groups["low"], groups["high"], self.target_size)
        image = np.asarray(Image.fromarray(crops[1]).resize((self.target_size[1], self.target_size[0]), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        return {"probability": probability.numpy(), "prediction": probability.argmax(0).numpy().astype(np.uint8), "image": image, "bbox_xyxy": box}


def predict_volume(config_path, input_path, output, device="auto"):
    """Predict each MRI slice and restore ROI labels to the input image grid.

    Args:
        config_path: 2.5D ensemble configuration JSON.
        input_path: NPY volume [N,H,W] or a single-series DICOM directory.
        output: Prediction output directory.
        device: auto, cpu, or a CUDA device.

    Returns:
        A summary containing slice count and the segmentation.npy path.

    Notes:
        Saves full-grid labels [N,H,W], model-resolution probabilities, and ROI
        coordinates for each slice. Resizes probabilities to the ROI before argmax.
        Pixels outside the ROI are background. NPY outputs carry no physical metadata.
    """
    predictor = EnsemblePredictor(config_path, device)
    volume = load_volume(input_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    labels = np.zeros(volume.shape, dtype=np.uint8)
    rows = []
    for index in range(len(volume)):
        result = predictor.predict_slice(volume, index)
        x1, y1, x2, y2 = result["bbox_xyxy"]
        # Restore probabilities before argmax, then paste the ROI into the original image grid.
        original_prob = F.interpolate(torch.from_numpy(result["probability"])[None], size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)[0]
        labels[index, y1:y2, x1:x2] = original_prob.argmax(0).numpy().astype(np.uint8)
        filename = f"slice_{index:04d}_prob.npy"
        np.save(output / filename, result["probability"])
        rows.append({"slice_index": index, "probability": filename, "bbox_xyxy": list(result["bbox_xyxy"])})
    np.save(output / "segmentation.npy", labels)
    write_json(output / "prediction.json", {"input": str(input_path), "shape": list(volume.shape), "samples": rows, "coordinates": "array [slice,y,x]; original image grid; outside ROI is background"})
    return {"slices": len(volume), "segmentation": str(output / "segmentation.npy")}
