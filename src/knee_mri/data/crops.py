"""One highest-confidence box, shared by each central image and its label."""
from pathlib import Path
import numpy as np
from PIL import Image
from .splits import parse_slice, SLICE_PATTERN
from ..common import require_artifact, write_json


def normalize_uint8(image):
    """Convert a finite two-dimensional grayscale image to uint8.

    Args:
        image: Grayscale array [H,W].

    Returns:
        A uint8 image. Values in [0,1] are multiplied by 255, values in [0,255]
        are rounded directly, and other ranges use min-max normalization.

    Raises:
        ValueError: Input is not two-dimensional or contains nonfinite values.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite grayscale image [H,W]")
    lo, hi = float(image.min()), float(image.max())
    # Preserve input intensity scaling if it is already a byte image or [0,1].
    if 0 <= lo and hi <= 1:
        return (image * 255).round().astype(np.uint8)
    if 0 <= lo and hi <= 255:
        return image.round().astype(np.uint8)
    return ((image - lo) / max(hi - lo, 1e-8) * 255).round().astype(np.uint8)


def best_bbox(image, detector):
    """Return the highest-confidence YOLO box clipped to image bounds.

    Args:
        image: Grayscale array [H,W], converted internally to three-channel uint8.
        detector: A YOLO detector exposing a predict method.

    Returns:
        Integer coordinates (x1,y1,x2,y2), with exclusive right/bottom endpoints.
        Returns None if no detection exists or the clipped box is empty.
    """
    rgb = np.repeat(normalize_uint8(image)[..., None], 3, axis=2)
    result = detector.predict(rgb, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    index = int(result.boxes.conf.argmax().item())
    x1, y1, x2, y2 = result.boxes.xyxy[index].cpu().numpy().astype(int)
    height, width = image.shape
    box = (max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2)))
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    return box


def crop_dataset(image_dir, annotation_dir, checkpoint, output, slice_start=1, mask_axes=(0, 1, 2)):
    """Crop slice images and NIfTI labels using the same detected boxes.

    Args:
        image_dir: Flat images named <study>_slice_001.jpg.
        annotation_dir: <study>.nii.gz labels read with SimpleITK.
        checkpoint: YOLO localization weight path.
        output: Output root; images and masks subdirectories must not exist.
        slice_start: Filename numbering origin used to index the label depth axis.
        mask_axes: Axis permutation converting SimpleITK arrays to [slice,y,x].

    Returns:
        Counts of cropped and undetected slices. Also saves images, NPY masks,
        and crops.json.

    Raises:
        OSError: Required files cannot be read or written.
        FileExistsError: Output sample directories already exist.
        ValueError: Axes, image/mask alignment, or integer labels are invalid.
    """
    from ultralytics import YOLO
    import SimpleITK as sitk
    if sorted(mask_axes) != [0, 1, 2]:
        raise ValueError("mask_axes must be a permutation of 0,1,2")
    detector = YOLO(str(require_artifact(checkpoint)))
    output = Path(output)
    if (output / "images").exists() or (output / "masks").exists():
        raise FileExistsError("Use a fresh crop output directory to avoid stale image/label pairs")
    images = sorted(p for p in Path(image_dir).iterdir() if p.is_file() and SLICE_PATTERN.fullmatch(p.name))
    if not images:
        raise ValueError("No <study>_slice_001.jpg images found")
    rows, skipped = [], []
    current_study, volume = None, None
    for path in images:
        study, number = parse_slice(path.name)
        if study != current_study:
            annotation = Path(annotation_dir) / f"{study}.nii.gz"
            volume = sitk.GetArrayFromImage(sitk.ReadImage(str(annotation))).transpose(mask_axes)
            current_study = study
        with Image.open(path) as img:
            image = np.asarray(img.convert("L"))
        z = number - slice_start
        if not 0 <= z < volume.shape[0] or tuple(volume.shape[1:]) != image.shape:
            raise ValueError(f"Image/annotation alignment mismatch: {path.name}; verify slice_start and mask_axes")
        box = best_bbox(image, detector)
        if box is None:
            skipped.append(path.name)
            continue
        x1, y1, x2, y2 = box
        label = volume[z, y1:y2, x1:x2]
        if not np.isfinite(label).all() or not np.equal(label, np.floor(label)).all() or label.min() < 0 or label.max() > 4:
            raise ValueError(f"Labels must be integers 0..4: {path.name}")
        image_path = output / "images" / path.name
        mask_path = output / "masks" / study / (path.stem + ".npy")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image[y1:y2, x1:x2]).save(image_path)
        np.save(mask_path, label.astype(np.uint8))
        rows.append({"sample": path.name, "bbox_xyxy": list(box), "slice_index": z})
    write_json(output / "crops.json", {"samples": rows, "no_detection": skipped, "slice_start": slice_start, "mask_axes": list(mask_axes), "mask_reader": "SimpleITK.GetArrayFromImage"})
    return {"cropped": len(rows), "no_detection": len(skipped)}
