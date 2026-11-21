"""NPY and single-series DICOM volume readers."""
import numpy as np


def validate_volume(volume):
    """Convert an input to a nonempty, finite three-dimensional float32 array.

    Args:
        volume: Array-like input with shape [N_slices,H,W].

    Returns:
        A float32 NumPy array without reordering axes or changing the intensity range.

    Raises:
        ValueError: The array is not three-dimensional, has an empty axis,
            or contains NaN or infinite values.
    """
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3 or any(n < 1 for n in volume.shape) or not np.isfinite(volume).all():
        raise ValueError("Expected finite volume [N_slices,H,W]")
    return volume


def load_dicom_series(files):
    """Read a single-frame grayscale DICOM series and normalize each slice.

    Args:
        files: Sequence of paths or file objects readable by pydicom.

    Returns:
        A float32 volume [N,H,W] sorted by InstanceNumber. RescaleSlope and
        RescaleIntercept are applied before per-slice normalization to [0,1].

    Raises:
        ValueError: Input is empty, mixes series, contains multiframe or color
            images, or has duplicate instance numbers.

    Notes:
        Missing InstanceNumber values fall back to input order. The caller must
        verify anatomical slice orientation.
    """
    from pydicom import dcmread
    records = []
    series = set()
    for order, file in enumerate(files):
        ds = dcmread(file)
        series.add(str(getattr(ds, "SeriesInstanceUID", "unknown")))
        image = ds.pixel_array.astype(np.float32)
        if image.ndim != 2:
            raise ValueError("Only single-frame grayscale DICOM slices are supported")
        image = image * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        image = (image - image.min()) / max(float(image.max() - image.min()), 1e-8)
        records.append((int(getattr(ds, "InstanceNumber", order)), image))
    if len(series) != 1:
        raise ValueError("Select files from exactly one DICOM SeriesInstanceUID")
    if len({r[0] for r in records}) != len(records):
        raise ValueError("Duplicate DICOM InstanceNumber values; verify slice order")
    return validate_volume(np.stack([r[1] for r in sorted(records, key=lambda r: r[0])]))


def load_volume(path):
    """Read an unlabeled MRI volume from an NPY file or DICOM directory.

    Args:
        path: An NPY file containing [N,H,W], or a directory containing .dcm files.

    Returns:
        A float32 volume validated by validate_volume.

    Raises:
        ValueError: No DICOM files exist or the volume format is invalid.

    Notes:
        NPY loading disables pickle and preserves the input intensity range.
    """
    from pathlib import Path
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.dcm"))
        if not files:
            raise ValueError(f"No .dcm files in {path}")
        return load_dicom_series(files)
    return validate_volume(np.load(path, allow_pickle=False))
