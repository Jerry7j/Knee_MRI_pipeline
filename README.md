# Knee MRI Pipeline

[English](README.md) | [简体中文](README.zh-CN.md)

A Python pipeline for knee MRI cartilage and meniscus segmentation, with preprocessing, 2.5D/3D model training, independent evaluation, multi-resolution fusion, visualization, and interactive inference.

## Research collaboration

This project analyzes knee MRI data provided by Shanghai Longhua Hospital as part of a research collaboration. The analysis focuses on cartilage and meniscus segmentation.

## Implementation references

This repository's implementation draws on the open-source project [pitthexai/Knee_MRI_Segmentation_2.5D](https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D) and the paper [KneeXNet-2.5D: a clinically-oriented and explainable deep learning framework for MRI-based knee cartilage and meniscus segmentation](https://www.nature.com/articles/s44401-026-00072-5), published in *npj Health Systems*. We thank the authors for sharing their research methods and code.

## Pipeline overview

The figure illustrates the reference KneeXNet-2.5D workflow: annotation and YOLOv11 localization, three-slice inputs at multiple resolutions and Gaussian scales, U-Net segmentation, ensemble prediction, and entropy-based visualization.

[![KneeXNet-2.5D pipeline overview](pipeline.png)](pipeline.png)

*Figure source: [pitthexai/Knee_MRI_Segmentation_2.5D — Figures/pipeline.png](https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D/blob/main/Figures/pipeline.png). Click the figure to view it at full size.*

## Project structure

```text
Knee_MRI_pipeline/
+-- configs/                   # Experiment and ensemble configurations
+-- src/knee_mri/
|   +-- data/                  # Datasets, patient splits, localization, and crops
|   +-- models/                # 2.5D/3D models and weight loading
|   +-- engine/                # Training, evaluation, prediction, metrics, and fusion
|   +-- visualization/         # Segmentation panels, entropy maps, and learning curves
|   +-- common.py              # Configuration, IO, devices, and random seeds
|   +-- cli.py                 # Unified command-line entry point
+-- apps/                      # Interactive Streamlit application
+-- tests/                     # Configuration, data, and numerical tests
+-- data/                      # Raw data, processed data, and dataset splits
+-- outputs/                   # Models, metrics, probabilities, and visualizations
+-- .python-version            # Python version for uv
+-- uv.lock                    # Resolved dependencies
+-- pyproject.toml
+-- requirements.txt
```

The four core modules are [data](src/knee_mri/data/README.md), [models](src/knee_mri/models/README.md), [execution](src/knee_mri/engine/README.md), and [visualization](src/knee_mri/visualization/README.md). All Python functions have English docstrings available through editor tooltips or `help()`.

## Installation

### Environment and dependency inventory

| Component | Requirement |
| --- | --- |
| Python | Package range: `>=3.10,<3.13`; uv default: **3.11** via `.python-version` |
| uv | Lockfile generated and checked with **0.13.9**; use a current uv release |
| Default backend | PyPI wheels for PyTorch **2.6.0** and torchvision **0.21.0** |
| NVIDIA backend | Optional local install: PyTorch **2.6.0+cu124**, torchvision **0.21.0+cu124**; Windows/Linux x86_64 with a compatible NVIDIA driver |
| Build backend | `setuptools>=68`, `wheel` (installed by the build frontend when needed) |

The following table lists every direct runtime and optional dependency from `pyproject.toml`. The resolved versions shown are for **Python 3.11 on Windows**, based on the lockfile installation plan. Other supported Python/platform combinations can resolve to different entries recorded in `uv.lock`.

| Feature | Package | Declared requirement | Locked version | Purpose |
| --- | --- | --- | --- | --- |
| Core | numpy | `>=1.24,<3` | 2.4.6 | Slice arrays, labels, probabilities, NPY files |
| Core | Pillow | `>=10` | 13.0.0 | JPEG/PNG image IO |
| Core | torch | `==2.6.0` | 2.6.0 + selected backend | Training, tensor operations, model inference |
| Core | torchvision | `==0.21.0` | 0.21.0 + selected backend | Image transforms and model utilities |
| Core | segmentation-models-pytorch | `>=0.4,<0.6` | 0.5.0 | Existing ResNet34 U-Net |
| Core | matplotlib | `>=3.7` | 3.11.2 | Segmentation panels and learning curves |
| Core | tqdm | `>=4.65` | 4.70.1 | Training/evaluation progress |
| `volume` | monai | `>=1.3,<2` | 1.5.2 | 3D U-Net |
| `volume` | nibabel | `>=5` | 5.5.0 | NIfTI image/label IO |
| `volume`, `localization` | SimpleITK | `>=2.3` | 2.5.8 | DICOM series and volume IO |
| `localization`, `app` | ultralytics | `>=8.3,<9` | 8.4.216 | YOLO ROI localization |
| `localization` | PyYAML | `>=6` | 6.0.4 | YOLO dataset configuration |
| `app` | streamlit | `>=1.40,<2` | 1.67.0 | Interactive inference UI |
| `app` | pydicom | `>=3` | 3.0.2 | DICOM metadata inspection |
| `app` | opencv-python | `>=4.8` | 5.1.0.95 | Image processing and application display |
| `dev` | pytest | `>=7` | 9.2.0 | Optional test runner; unittest is in Python's standard library |

`uv.lock` also records indirect dependencies and wheel hashes, including timm, huggingface-hub, safetensors, pandas, pyarrow, and the NVIDIA runtime packages selected on Linux. uv installs them through their parent packages. PFDNet and Res2Net are included in this repository and use PyTorch; no separate PFDNet package is required.

For DICOM series inference without the UI, include `localization` (SimpleITK); for NIfTI/3D processing include `volume`. Model `.pt`/`.pth` files and MRI data are separate inputs and are not downloaded by uv. A system CUDA toolkit is not required for these prebuilt PyTorch wheels. CPU use requires no CUDA installation. Use CUDA 12.4 only on a compatible NVIDIA system; macOS Apple Silicon uses PyPI wheels and has not been validated here.

The uv lockfile has been checked. This configuration update does not install a new environment; the complete locked dependency set has not been run through the application tests.

### Using uv (recommended)

Only three files define the environment: `pyproject.toml` lists dependencies, `.python-version` selects Python 3.11, and `uv.lock` records resolved versions. No separate uv configuration is needed. Run these commands from the project directory when you want to install the environment:

```powershell
python -m pip install --upgrade uv
uv python install 3.11
uv sync --locked --extra volume --extra localization --extra app --extra dev
uv run --no-sync knee-mri check
```

For only 2.5D training, use `uv sync --locked`. To run commands without activating `.venv`, prefix them with `uv run --no-sync`:

```powershell
uv run --no-sync knee-mri train --config configs/pfdnet_25d.json
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync streamlit run apps/streamlit_app.py
```

The default installation uses PyPI wheels; CUDA availability depends on the platform and installed PyTorch build. If you need the CUDA 12.4 build on Windows/Linux x86_64, run this optional command **after** syncing:

```powershell
uv pip install --reinstall-package torch --reinstall-package torchvision torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv run --no-sync python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

This CUDA override is a local choice and is not locked by `uv.lock`. Continue using `uv run --no-sync` to preserve it; after another `uv sync`, reapply the CUDA command if needed. A compatible NVIDIA driver is required; a system CUDA toolkit is not needed for the prebuilt wheels.

After changing dependencies, run `uv lock` and commit the updated lockfile with `pyproject.toml`. `.venv` stays local. This update only creates configuration and documentation; it does not install an environment.

### Using pip

Use Python 3.10 or 3.11 and run these commands from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[volume,localization,app,dev]"
knee-mri --help
knee-mri check
```

For 2.5D training and evaluation only, install with `pip install -e .`. The `volume` extra enables 3D models, `localization` enables YOLO, and `app` enables the interactive application. `requirements.txt` installs all features. Choose a GPU build of PyTorch compatible with your CUDA environment.

## Data format

Pair 2.5D images and masks using this layout:

```text
data/processed/
+-- images/
|   +-- 9001104_20050825_10498205_slice_001.jpg
+-- masks/
    +-- 9001104_20050825_10498205/
        +-- 9001104_20050825_10498205_slice_001.npy
```

Images are grayscale JPEGs, and masks are two-dimensional integer arrays with matching dimensions. Three neighboring slices form a `[3,H,W]` input; missing neighbors repeat the central slice. Class definitions:

| ID | Class |
| --- | --- |
| 0 | Background |
| 1 | Distal femoral cartilage |
| 2 | Proximal tibial cartilage |
| 3 | Patellar cartilage |
| 4 | Meniscus |

3D data uses `data/processed/volumes/<study>/*.jpg` and `data/raw/annotations/<study>.nii.gz`. Slice order, mask orientation, and pixel locations must align. `mask_axis_order` controls the mask array axis permutation.

The patient ID is the study-name prefix before the first underscore. All visits from one patient remain in the same subset. Experiments at different resolutions share the same split.

## Training and evaluation

Set the data paths in the [experiment configurations](configs/README.md) first. Relative configuration paths resolve against `project_root`, which defaults to this project root. Additional command-line paths resolve against the current working directory.

```powershell
# Create patient-level splits
knee-mri split --config configs/noise_256_k5_s015.json

# Train and validate
knee-mri train --config configs/noise_256_k5_s015.json

# Evaluate the best model independently
knee-mri evaluate --config configs/noise_256_k5_s015.json

# Save segmentation, ground-truth, and entropy panels
knee-mri visualize --config configs/noise_256_k5_s015.json --exports outputs/noise_256_k5_s015/test --output outputs/noise_256_k5_s015/figures

# Save learning curves
knee-mri curves --log outputs/noise_256_k5_s015/training_log.csv --output outputs/noise_256_k5_s015/curves.png
```

Available configurations include a variable-size 2.5D baseline, four blur experiments at 256/512 resolution, and a 3D U-Net. The 3D workflow uses separate data directories and a volume split:

```powershell
knee-mri split --config configs/unet3d.json
knee-mri train --config configs/unet3d.json
knee-mri evaluate --config configs/unet3d.json
```

Training saves `best_model.pth`, `best.ckpt`, `last.ckpt`, CSV logs, and a JSON summary. Validation IoU selects the best model. Reusing an output directory overwrites existing training artifacts.

Evaluation exports `metrics.json`, a per-sample `manifest.json`, predicted labels, and softmax probabilities. Use `--checkpoint <path>` to select weights, `--subset val` for validation, and `--no-probs` to disable probability exports. Dice/IoU are macro-averaged over classes 1 through 4, then averaged across samples. A class absent from both prediction and target receives a score of one.

## PFDNet segmentation

The author's [PFDNet implementation](src/knee_mri/models/pfdnet.py) is included with its Res2Net encoder and two parallel fusion decoding paths. Its output head now predicts five segmentation logits: background, femoral cartilage, tibial cartilage, patellar cartilage, and meniscus. Inputs remain three neighbouring MRI slices in the existing 2.5D pipeline.

Set `"model_type": "2.5d"`, `"architecture": "pfdnet"`, and `"pfdnet_channels": 32` in the experiment configuration. The last option controls decoder width and must match during training and inference. Existing configurations default to U-Net. PFDNet also supports the variable-size workflow with padding to multiples of 32.

```powershell
knee-mri split --config configs/pfdnet_25d.json
knee-mri train --config configs/pfdnet_25d.json
knee-mri evaluate --config configs/pfdnet_25d.json
knee-mri predict --config configs/ensemble_pfdnet.json --input data/raw/volume.npy --output outputs/pfdnet_inference
```

Configure the data paths and train MRI segmentation weights before prediction. The existing landmark weights and downloaded U-Net weights cannot be directly loaded into this five-class PFDNet model. No PFDNet pretrained weights are downloaded automatically. The original YOLO localizer is retained. Each model in the ensemble can select unet or pfdnet and use the existing resolution-group fusion.

### PFDNet paper

When using the PFDNet architecture, cite the author's paper, credit the paper and code sources, and retain the applicable copyright and attribution notices.

Wu Z, Zhang R, Bai H, Ma J, Ma X, Zhu X. [Keypoint estimation of knee x-ray images using a parallel fusion decoding network](https://www.sciencedirect.com/science/article/abs/pii/S0968016022002101). *The Knee*.

## Multi-resolution fusion

Train and evaluate each of the four noise configurations, then run:

```powershell
knee-mri fuse --config configs/noise_512_k5_s015.json --low outputs/noise_256_k5_s015/test outputs/noise_256_k7_s010/test --high outputs/noise_512_k5_s015/test outputs/noise_512_k7_s010/test --output outputs/fusion
knee-mri visualize --config configs/noise_512_k5_s015.json --exports outputs/fusion --output outputs/fusion/figures --mask-background
```

Probability maps are bilinearly resized to the target size, averaged within each resolution group, then averaged equally across groups. Inputs must be fixed-size 2.5D exports from the same patient split and crop version. Fusion validates sample IDs, order, and data directories.

## YOLO localization and cropping

```powershell
knee-mri yolo-prepare --images data/raw/images --labels data/raw/yolo_labels --output data/processed/yolo
knee-mri yolo-train --data data/processed/yolo/KneeMRI.yaml --checkpoint yolo11n.pt --epochs 50 --output outputs/yolo
knee-mri yolo-evaluate --data data/processed/yolo/KneeMRI.yaml --checkpoint ../Models/localizer.pt --output outputs/yolo_test
knee-mri crop --images data/raw/images --annotations data/raw/annotations --checkpoint ../Models/localizer.pt --output data/processed --slice-start 1 --mask-axes 0 1 2
```

YOLO uses matching txt labels with one `0 x_center y_center width height` detection per line and coordinates normalized to [0,1]. Negative samples require empty txt files. The highest-confidence box crops both image and mask. Use fresh output directories for dataset preparation and cropping.

Cropping reads masks with SimpleITK, so `--mask-axes` applies to its array order. The 3D dataset reads masks with nibabel; verify axis conventions separately for each reader.

## Unlabeled inference and application

Set model paths, resolution groups, and the optional localizer in [ensemble.json](configs/ensemble.json). Default weight paths point to `../Models`; you can also use `outputs/<experiment>/best_model.pth`. Set `localizer` to null for inputs that are already cropped and aligned.

```powershell
knee-mri predict --config configs/ensemble.json --input data/raw/volume.npy --output outputs/inference
streamlit run apps/streamlit_app.py
```

NPY input has shape `[N,H,W]`. A DICOM directory must contain single-frame grayscale slices from one series, sorted by InstanceNumber. Inference applies the central slice's bounding box to all three neighboring slices and reports an error if no ROI is detected.

The CLI saves `segmentation.npy` on the input image grid, per-slice probabilities, and ROI coordinate records. The application displays a selected slice and downloads its ROI segmentation. NPY outputs do not contain NIfTI affine or DICOM spatial metadata.

## Tests

```powershell
python -m unittest discover -s tests -v
```

See [tests/README.md](tests/README.md) for coverage. Tests requiring unavailable scientific dependencies are skipped. Full training and inference require imaging data and actual model weights. Git LFS pointers cannot be loaded as weights; use `knee-mri check` to inspect artifact status.
