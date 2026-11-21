"""2.5D context stacking with resizing, padding, and optional blur."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import GaussianBlur
from .splits import parse_slice


def read_gray(path):
    """Read an image as a normalized two-dimensional grayscale array.

    Args:
        path: Image path readable by Pillow.

    Returns:
        A float32 array [H,W] with intensities divided by 255 to obtain [0,1].
    """
    with Image.open(path) as image:
        return np.array(image.convert("L"), dtype=np.float32) / 255.0


def center_pad(array, shape):
    """Center a two-dimensional array within a larger zero-padded canvas.

    Args:
        array: Image or label array [H,W].
        shape: Target (height, width), with neither axis smaller than the input.

    Returns:
        A padded NumPy array. Odd padding differences add one extra pixel to the
        bottom or right side.

    Raises:
        ValueError: The target is smaller than the input.
    """
    h, w = shape[0] - array.shape[0], shape[1] - array.shape[1]
    if h < 0 or w < 0:
        raise ValueError("Cannot pad to a smaller shape")
    return np.pad(array, ((h // 2, h - h // 2), (w // 2, w - w // 2)))


class KneeSegmentation25D(Dataset):
    def __init__(self, image_dir, mask_dir, filenames, input_size=(256, 256), resize=True, blur_kernel=0, blur_sigma=0.15):
        """Initialize a three-slice segmentation dataset with optional image blur.

        Args:
            image_dir: Flat directory of slice images.
            mask_dir: Root containing NPY masks in study subdirectories.
            filenames: Central-slice filenames, sorted internally.
            input_size: Target (height, width) when resizing is enabled.
            resize: If False, center-pad neighbors to their largest spatial dimensions.
            blur_kernel: Gaussian kernel size; zero disables blur.
            blur_sigma: Gaussian standard deviation applied to images only.
        """
        self.image_dir, self.mask_dir = Path(image_dir), Path(mask_dir)
        self.filenames = sorted(filenames)
        self.input_size, self.resize = tuple(input_size), resize
        self.blur = GaussianBlur(blur_kernel, sigma=blur_sigma) if blur_kernel else None

    def __len__(self):
        """Return the number of samples available as central slices."""
        return len(self.filenames)

    def __getitem__(self, index):
        """Load three neighboring images and the central label, validating alignment.

        Args:
            index: Index in the sorted central-slice filename list.

        Returns:
            An (image, mask) tuple of float32 [3,H,W] and int64 [H,W] tensors.
            Missing neighbors repeat the center image. Labels use nearest-neighbor
            resizing or zero padding only.

        Raises:
            ValueError: Image and label sizes differ, or labels are not integers in 0..4.
        """
        filename = self.filenames[index]
        study, number = parse_slice(filename)
        current_path = self.image_dir / filename
        # Reuse the central image if its immediate neighbor was not detected/cropped.
        suffix = Path(filename).suffix
        digits = len(Path(filename).stem.rsplit("_slice_", 1)[1])
        images = []
        for offset in (-1, 0, 1):
            path = self.image_dir / f"{study}_slice_{number + offset:0{digits}d}{suffix}"
            images.append(read_gray(path if path.is_file() else current_path))
        mask = np.load(self.mask_dir / study / (Path(filename).stem + ".npy"), allow_pickle=False)
        if mask.ndim != 2 or mask.shape != images[1].shape:
            raise ValueError(f"Image/mask alignment mismatch: {filename}")
        if not np.isfinite(mask).all() or not np.equal(mask, np.floor(mask)).all() or mask.min() < 0 or mask.max() > 4:
            raise ValueError(f"Mask must contain integer labels 0..4: {filename}")
        if self.resize:
            # PIL bilinear image resize matches the original torchvision/PIL preprocessing.
            size = (self.input_size[1], self.input_size[0])
            images = [np.asarray(Image.fromarray((im * 255).round().astype(np.uint8)).resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255 for im in images]
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(size, Image.Resampling.NEAREST)).copy()
        else:
            shape = np.max([im.shape for im in images], axis=0)
            images = [center_pad(im, shape) for im in images]
            mask = center_pad(mask, shape)
        tensor = torch.from_numpy(np.stack(images).astype(np.float32))
        if self.blur is not None:
            tensor = self.blur(tensor)
        return tensor, torch.from_numpy(mask.astype(np.int64))


def pad_batch_to_32(batch):
    """Pad variable-size samples to common spatial dimensions divisible by 32.

    Args:
        batch: Sequence of (image [3,H,W], mask [H,W]) tensor pairs.

    Returns:
        Images [B,3,H_pad,W_pad] and masks [B,H_pad,W_pad].
        Bottom/right image padding is zero; label padding is -100 so loss and
        metric calculations can ignore it.
    """
    images, masks = zip(*batch)
    height = ((max(x.shape[-2] for x in images) + 31) // 32) * 32
    width = ((max(x.shape[-1] for x in images) + 31) // 32) * 32
    pads = [(0, width - x.shape[-1], 0, height - x.shape[-2]) for x in images]
    # Padding labels are ignored by loss/metrics rather than counted as background.
    return (torch.stack([F.pad(x, p) for x, p in zip(images, pads)]),
            torch.stack([F.pad(y, p, value=-100) for y, p in zip(masks, pads)]))
