"""Prevent adjacent-slice and repeat-visit patient leakage between subsets."""
import random
import re
from pathlib import Path
from ..common import read_json, write_json

SLICE_PATTERN = re.compile(r"^(?P<study>.+)_slice_(?P<index>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
SUBSETS = ("train", "val", "test")


def parse_slice(filename):
    """Extract a study identifier and slice number from a filename.

    Args:
        filename: A name such as <study>_slice_001.jpg, with jpeg or png also accepted.

    Returns:
        A (study_id, slice_index) tuple with an integer slice index.

    Raises:
        ValueError: The filename does not match the slice naming convention.
    """
    match = SLICE_PATTERN.fullmatch(filename)
    if not match:
        raise ValueError(f"Expected <study>_slice_001.jpg, got {filename!r}")
    return match.group("study"), int(match.group("index"))


def patient_id(sample, volume=False):
    """Extract the patient identifier used for grouped dataset splitting.

    Args:
        sample: A slice filename, or a study directory name when volume is True.
        volume: Whether sample directly identifies a volume study.

    Returns:
        The study prefix before the first underscore, grouping repeated visits
        from the same patient together.
    """
    study = sample if volume else parse_slice(sample)[0]
    return study.split("_", 1)[0]


def validate_split(split, volume=False):
    """Validate nonempty subsets, unique samples, and patient separation.

    Args:
        split: Dictionary of train, val, and test filename or study-name lists.
        volume: Whether entries identify volume studies rather than slice files.

    Returns:
        The validated split dictionary without modifying its contents.

    Raises:
        ValueError: A subset is empty, a name contains a path, samples are duplicated,
            or a patient appears in multiple subsets.
    """
    groups = {}
    for subset in SUBSETS:
        samples = split.get(subset)
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Split {subset!r} must be a nonempty list")
        if any(not isinstance(s, str) or Path(s).name != s or s in (".", "..") or "/" in s or "\\" in s for s in samples):
            raise ValueError("Split entries must be simple filenames or study IDs")
        if len(samples) != len(set(samples)):
            raise ValueError(f"Duplicate samples in {subset}")
        groups[subset] = {patient_id(s, volume) for s in samples}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[a] & groups[b]
        if overlap:
            raise ValueError(f"Patient leakage between {a}/{b}: {sorted(overlap)[:5]}")
    return split


def create_split(image_dir, output, volume=False, seed=42, val_fraction=0.1, test_fraction=0.1):
    """Create a patient-grouped train/validation/test split and save it.

    Args:
        image_dir: Flat slice directory, or a root of study directories for volumes.
        output: Output JSON path.
        volume: Whether to enumerate study subdirectories instead of slice files.
        seed: Local random seed used to shuffle patient identifiers.
        val_fraction: Fraction of patients assigned to validation.
        test_fraction: Fraction of patients assigned to testing.

    Returns:
        A dictionary of train, val, and test sample lists, also written to output.

    Raises:
        ValueError: Fractions are invalid, fewer than three patients exist,
            or the requested allocation cannot keep every subset nonempty.
    """
    root = Path(image_dir)
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    samples = sorted(p.name for p in root.iterdir() if p.is_dir()) if volume else sorted(
        p.name for p in root.iterdir() if p.is_file() and SLICE_PATTERN.fullmatch(p.name))
    patients = sorted({patient_id(s, volume) for s in samples})
    if len(patients) < 3:
        raise ValueError("At least three patients are needed for independent train/val/test splits")
    random.Random(seed).shuffle(patients)
    n_val = max(1, round(len(patients) * val_fraction))
    n_test = max(1, round(len(patients) * test_fraction))
    if n_val + n_test >= len(patients):
        raise ValueError("Too few patients for requested split fractions")
    assignment = {p: "val" if i < n_val else "test" if i < n_val + n_test else "train" for i, p in enumerate(patients)}
    split = {key: [s for s in samples if assignment[patient_id(s, volume)] == key] for key in SUBSETS}
    validate_split(split, volume)
    write_json(output, split)
    return split


def load_split(path, volume=False):
    """Read and validate a patient split JSON file.

    Args:
        path: Split file path.
        volume: Whether entries are study names rather than slice filenames.

    Returns:
        A split dictionary checked for empty subsets, duplicates, and patient leakage.
    """
    return validate_split(read_json(path), volume)
