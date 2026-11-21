"""YOLO detection dataset preparation, training, and evaluation."""
from pathlib import Path
import shutil
from .splits import create_split
from ..common import require_artifact, write_json


def prepare_yolo_dataset(image_dir, label_dir, output, seed=42):
    """Create a single-class YOLO detection dataset with patient-grouped splits.

    Args:
        image_dir: Flat images following the study/slice naming convention.
        label_dir: Matching txt labels; negative samples require empty label files.
        output: Root where train, val, and test directories must not already exist.
        seed: Random seed for patient splitting.

    Returns:
        The absolute KneeMRI.yaml path. Copies images and labels and saves split.json.

    Raises:
        FileExistsError: An output subset directory already exists.
        FileNotFoundError: A required detection label is missing.
        ValueError: A label violates the single-class normalized detection format.
    """
    import yaml
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / subset).exists() for subset in ("train", "val", "test")):
        raise FileExistsError("Use a new YOLO output directory to avoid mixing old and new splits")
    split = create_split(image_dir, output / "split.json", seed=seed)
    # Check every label before copying. An explicit empty file denotes a negative image.
    for samples in split.values():
        for sample in samples:
            label = Path(label_dir) / (Path(sample).stem + ".txt")
            if not label.is_file():
                raise FileNotFoundError(f"Missing YOLO label: {label}")
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Expected detection label '0 x y w h': {label}")
                values = [float(v) for v in fields]
                if values[0] != 0 or not all(0 <= v <= 1 for v in values[1:]) or min(values[3:]) <= 0:
                    raise ValueError(f"Invalid YOLO detection label: {label}")
    for subset, samples in split.items():
        for folder in ("images", "labels"):
            (output / subset / folder).mkdir(parents=True, exist_ok=True)
        for sample in samples:
            label = Path(sample).stem + ".txt"
            shutil.copy2(Path(image_dir) / sample, output / subset / "images" / sample)
            shutil.copy2(Path(label_dir) / label, output / subset / "labels" / label)
    info = {"path": str(output), "train": "train/images", "val": "val/images", "test": "test/images", "names": {0: "SagittalKneeJoint"}, "nc": 1}
    (output / "KneeMRI.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    return output / "KneeMRI.yaml"


def train_yolo(data_yaml, checkpoint="yolo11n.pt", epochs=50, output="outputs/yolo", device=None):
    """Train a YOLO localization model using the selected data configuration.

    Args:
        data_yaml: YOLO dataset configuration path.
        checkpoint: Initial weights or an Ultralytics model name, which may download.
        epochs: Number of training epochs.
        output: Training project output root.
        device: Optional Ultralytics device setting.

    Returns:
        The result returned by the Ultralytics training call.

    Raises:
        FileNotFoundError: The data YAML does not exist.
    """
    from ultralytics import YOLO
    if not Path(data_yaml).is_file():
        raise FileNotFoundError(data_yaml)
    kwargs = {"device": device} if device is not None else {}
    return YOLO(checkpoint).train(data=str(Path(data_yaml).resolve()), epochs=epochs, project=str(Path(output).resolve()), name="SagittalKneeJoint", **kwargs)


def evaluate_yolo(checkpoint, data_yaml, subset="test", output="outputs/yolo_eval"):
    """Evaluate a YOLO localization model on validation or test data.

    Args:
        checkpoint: Path to actual YOLO weights.
        data_yaml: Dataset configuration path.
        subset: val or test.
        output: Evaluation output root.

    Returns:
        The Ultralytics metrics dictionary. Numeric metrics are also saved as
        metrics.json.
    """
    from ultralytics import YOLO
    metrics = YOLO(str(require_artifact(checkpoint))).val(data=str(Path(data_yaml).resolve()), split=subset, project=str(Path(output).resolve()), name=subset)
    write_json(Path(output) / "metrics.json", {k: float(v) for k, v in metrics.results_dict.items()})
    return metrics.results_dict
