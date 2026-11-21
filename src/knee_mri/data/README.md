# Data and preprocessing

[English](README.md) | [简体中文](README.zh-CN.md)

MRI input readers, segmentation datasets, patient splits, and YOLO localization/cropping.

| File | Function |
| --- | --- |
| `splits.py` | Slice-name parsing, patient grouping, and JSON split generation/validation |
| `slices.py` | Three-slice 2.5D datasets, resizing, blur, and padding |
| `volumes.py` | JPEG volumes, NIfTI masks, and shared 3D spatial augmentation |
| `loaders.py` | Configuration-driven train/val/test loaders |
| `inputs.py` | NPY and single-series DICOM volume readers |
| `yolo.py` | Detection dataset preparation and YOLO training/evaluation |
| `crops.py` | Highest-confidence boxes and paired image/mask cropping |

## Segmentation data

2.5D samples use flat `<study>_slice_001.jpg` images and `<mask_dir>/<study>/<slice_stem>.npy` masks, returning float32 images `[3,H,W]` and int64 masks `[H,W]`. Images use bilinear resizing; masks use nearest-neighbor resizing. Missing neighbors repeat the center. Variable-size mode centers neighboring slices on a common canvas, then pads batches on the bottom/right to multiples of 32. Extra mask pixels are -100.

3D samples use `<image_dir>/<study>/*.jpg` and `<mask_dir>/<study>.nii.gz`, returning `[1,D,H,W]` images and `[D,H,W]` masks. `mask_axis_order` configures the nibabel mask axis permutation. Training images and masks share random spatial transforms; validation/test has no random augmentation. Shape mismatches or labels outside integer classes 0 through 4 raise errors.

```powershell
knee-mri split --config configs/noise_256_k5_s015.json
```

The patient ID is the first study-name segment. All visits from one patient stay in one subset. Train, val, and test must be nonempty, with no duplicate samples or cross-subset patient leakage.

## Localization and cropping

```powershell
knee-mri yolo-prepare --images data/raw/images --labels data/raw/yolo_labels --output data/processed/yolo
knee-mri yolo-train --data data/processed/yolo/KneeMRI.yaml --epochs 50 --output outputs/yolo
knee-mri yolo-evaluate --data data/processed/yolo/KneeMRI.yaml --checkpoint ../Models/localizer.pt
knee-mri crop --images data/raw/images --annotations data/raw/annotations --checkpoint ../Models/localizer.pt --output data/processed --slice-start 1 --mask-axes 0 1 2
```

YOLO txt labels use `0 x y w h` with normalized coordinates. Negative samples require empty txt files. Dataset preparation creates train/val/test images and labels plus KneeMRI.yaml.

Cropping reads masks with SimpleITK, selects the highest-confidence box for paired cropping, and saves images, masks, and crops.json. Undetected samples are recorded and skipped. Both operations require fresh sample output directories.

## Unlabeled input

`load_volume` accepts an `[N,H,W]` NPY file or a .dcm directory. DICOM input is limited to single-frame grayscale images from one series, sorted by InstanceNumber. Duplicate numbers or mixed series raise errors. The caller must verify slice orientation and mask alignment.
