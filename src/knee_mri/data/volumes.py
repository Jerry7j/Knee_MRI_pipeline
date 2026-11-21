"""3D comparator: study JPEG folders plus NIfTI masks."""
from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from .slices import read_gray


class KneeVolumeDataset3D(Dataset):
    def __init__(self, image_dir, mask_dir, study_ids, input_size=(112, 224, 224), training=False, mask_axis_order=(0, 1, 2)):
        """Initialize paired JPEG volumes, NIfTI masks, and shared spatial transforms.

        Args:
            image_dir: Root containing one JPEG subdirectory per study.
            mask_dir: Directory containing <study>.nii.gz labels.
            study_ids: Study identifiers, sorted internally.
            input_size: Target (depth, height, width).
            training: Enable shared random flips and affine transforms for image and mask.
            mask_axis_order: Axis permutation converting nibabel arrays to [D,H,W].

        Raises:
            ValueError: mask_axis_order is not a permutation of 0, 1, and 2.
        """
        from monai.transforms import Compose, Resized, ScaleIntensityd, RandFlipd, RandAffined
        self.image_dir, self.mask_dir = Path(image_dir), Path(mask_dir)
        self.filenames = sorted(study_ids)
        if sorted(mask_axis_order) != [0, 1, 2]:
            raise ValueError("mask_axis_order must be a permutation of [0,1,2]")
        self.mask_axis_order = tuple(mask_axis_order)
        steps = [Resized(keys=["image", "mask"], spatial_size=input_size, mode=("trilinear", "nearest")), ScaleIntensityd(keys="image")]
        if training:
            steps += [RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
                      RandAffined(keys=["image", "mask"], prob=0.3, rotate_range=(0.1,) * 3, translate_range=(5,) * 3, scale_range=(0.1,) * 3, mode=("bilinear", "nearest"))]
        # Dictionary transforms share the same spatial randomization for image and mask.
        self.transform = Compose(steps)

    def __len__(self):
        """Return the number of complete MRI volumes in the dataset."""
        return len(self.filenames)

    def __getitem__(self, index):
        """Stack slices in numeric order, read the mask, and apply shared 3D transforms.

        Args:
            index: Index of a study in the sorted study list.

        Returns:
            A float32 image tensor [1,D,H,W] and an int64 mask tensor [D,H,W].

        Raises:
            ValueError: Images are missing, filenames lack numeric slice indices,
                volume shapes differ, or labels are invalid.

        Notes:
            Uses only the configured axis permutation; does not infer orientation
            or silently truncate volume depth.
        """
        import nibabel as nib
        study = self.filenames[index]
        def slice_key(path):
            """Extract the trailing integer from a JPEG stem for slice ordering.

            Args:
                path: Path object identifying the slice image.

            Returns:
                The integer slice index.

            Raises:
                ValueError: The filename stem has no trailing index.
            """
            match = re.search(r"(\d+)$", path.stem)
            if not match:
                raise ValueError(f"Missing numeric slice index: {path}")
            return int(match.group(1))
        files = sorted((self.image_dir / study).glob("*.jpg"), key=slice_key)
        if not files:
            raise ValueError(f"No slices in {self.image_dir / study}")
        image = np.stack([read_gray(p) for p in files])
        mask = np.asarray(nib.load(self.mask_dir / f"{study}.nii.gz").dataobj).transpose(self.mask_axis_order).copy()
        if image.shape != mask.shape:
            raise ValueError(f"Volume alignment mismatch for {study}: {image.shape} vs {mask.shape}. Verify axis order and slice numbering.")
        if not np.isfinite(mask).all() or not np.equal(mask, np.floor(mask)).all() or mask.min() < 0 or mask.max() > 4:
            raise ValueError(f"Invalid label IDs in {study}")
        result = self.transform({"image": image[None].astype(np.float32), "mask": mask[None].astype(np.float32)})
        return torch.as_tensor(result["image"]).float(), torch.as_tensor(result["mask"][0]).long()
