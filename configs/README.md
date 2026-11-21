# Experiment configurations

[English](README.md) | [简体中文](README.zh-CN.md)

| File | Settings |
| --- | --- |
| `baseline_25d.json` | Variable-size 2.5D, three neighboring slices, center and batch padding, no blur |
| `noise_256_k5_s015.json` | 256×256，GaussianBlur kernel=5、sigma=0.15 |
| `noise_256_k7_s010.json` | 256×256，kernel=7、sigma=0.10 |
| `noise_512_k5_s015.json` | 512×512，kernel=5、sigma=0.15 |
| `noise_512_k7_s010.json` | 512×512，kernel=7、sigma=0.10 |
| `unet3d.json` | MONAI 3D U-Net，112×224×224，2 epoch |
| `ensemble.json` | Segmentation models, resolution groups, and optional localizer |

Training configurations define data and output paths through `image_dir`, `mask_dir`, `split_path`, and `output_dir`. Relative paths resolve against `project_root`. The four noise configurations share `patient_split.json`, with defaults of 30 epochs, batch=8, Adam lr=1e-4, and seed=65.

`device=auto` selects CPU/CUDA automatically; Windows defaults to `num_workers=0`. `pretrained=false` disables ImageNet downloads; true enables pretrained 2.5D encoders. With `resize=false`, `input_size` does not resize images.

For 3D data, `mask_axis_order` permutes the nibabel mask array into [D,H,W]. Set it to match the dataset.

Each ensemble model specifies `checkpoint`, `size`, and `group` (low/high). Probabilities are averaged within groups, then equally across groups. Checkpoint and localizer paths default to `../Models` and can point to training outputs instead. Use `localizer=null` for cropped, aligned volumes. Training and ensemble configurations are not interchangeable.
