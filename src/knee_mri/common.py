"""Configuration, JSON IO, artifact validation, and runtime setup."""
import json
import random
from pathlib import Path


def read_json(path):
    """Read a UTF-8 JSON file, accepting an optional byte-order mark.

    Args:
        path: Path to the JSON file.

    Returns:
        The decoded Python object, usually a configuration dictionary.

    Raises:
        OSError: The file cannot be read.
        ValueError: The file does not contain valid JSON.
    """
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    """Write an object as UTF-8 JSON and create missing parent directories.

    Args:
        path: Output file path; an existing file is overwritten.
        value: JSON-serializable object. Unicode is preserved with two-space indentation.

    Raises:
        TypeError: The object cannot be serialized.
        ValueError: The object contains NaN or infinite values.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require_artifact(path):
    """Check that an artifact exists and is not a Git LFS pointer.

    Args:
        path: Path to a model weight file or another artifact.

    Returns:
        The validated pathlib.Path. File format and model compatibility are not checked.

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: The file contains a Git LFS pointer instead of the artifact.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with path.open("rb") as handle:
        if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{path} is a Git LFS pointer. Fetch the actual weights with git lfs pull in the repository containing the model files.")
    return path


def validate_architecture(config):
    """Validate model selection without importing scientific dependencies."""
    architecture = config.get("architecture", "unet")
    if architecture not in ("unet", "pfdnet"):
        raise ValueError("architecture must be 'unet' or 'pfdnet'")
    if architecture == "pfdnet":
        if config.get("model_type") != "2.5d":
            raise ValueError("PFDNet supports model_type '2.5d' only")
        width = config.get("pfdnet_channels", 128)
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("pfdnet_channels must be a positive integer")
        if config.get("pretrained", False):
            raise ValueError("PFDNet does not download pretrained weights; use pretrained=false")
    return architecture


def load_config(path):
    """Load an experiment configuration, resolve paths, and validate settings.

    Args:
        path: Training or evaluation JSON path, not an ensemble configuration.

    Returns:
        A configuration dictionary with absolute data, split, checkpoint, and output
        paths. Relative paths use project_root relative to the configuration directory.

    Raises:
        ValueError: The model type, class count, dimensions, or training settings
            do not satisfy the pipeline requirements.
    """
    path = Path(path).resolve()
    config = read_json(path)
    root = (path.parent / config.get("project_root", "..")).resolve()
    for key in ("image_dir", "mask_dir", "split_path", "output_dir", "checkpoint"):
        if config.get(key):
            config[key] = str((root / config[key]).resolve())
    if config.get("model_type") not in ("2.5d", "3d"):
        raise ValueError("model_type must be '2.5d' or '3d'")
    validate_architecture(config)
    size = config.get("input_size", [])
    dimensions = 3 if config["model_type"] == "3d" else 2
    if len(size) != dimensions or any(not isinstance(n, int) or n <= 0 for n in size):
        raise ValueError(f"input_size must contain {dimensions} positive integers")
    divisor = 16 if dimensions == 3 else 32
    if config.get("resize", True) and any(n % divisor for n in size):
        raise ValueError(f"input_size must be divisible by {divisor}")
    if config.get("num_classes", 5) != 5:
        raise ValueError("This pipeline and its palette require five classes")
    for key in ("epochs", "batch_size"):
        if config.get(key, 1) < 1:
            raise ValueError(f"{key} must be positive")
    return config


def seed_everything(seed=65):
    """Set random seeds and deterministic cuDNN options for training.

    Args:
        seed: Shared integer seed for Python, NumPy, and CPU/CUDA PyTorch.

    Notes:
        Updates process-wide random states and cuDNN settings. Bitwise agreement
        across different hardware or dependency versions is not guaranteed.
    """
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name="auto"):
    """Select a PyTorch compute device from the configured name.

    Args:
        name: auto, cpu, or a device such as cuda:0. Auto prefers available CUDA.

    Returns:
        A torch.device object. Explicit device availability is checked at runtime.
    """
    import torch
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
