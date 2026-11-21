# knee_mri package

[English](README.md) | [简体中文](README.zh-CN.md)

Four modules organize the MRI segmentation workflow through a shared command-line interface:

| Module | Responsibility |
| --- | --- |
| `data` | Input readers, datasets, patient splits, YOLO localization, and paired crops |
| `models` | 2.5D/3D U-Net construction and weight loading |
| `engine` | Training, validation, evaluation, inference, metrics, and probability fusion |
| `visualization` | Segmentation comparisons, overlays, entropy maps, and learning curves |

`common.py` provides JSON IO, configuration validation, artifact checks, random seeds, and device selection. Relative paths resolve against the configured `project_root`; JSON writes create parent directories. Auto device selection prefers cuda:0 when available, otherwise CPU.

`cli.py` provides 12 commands. `python -m knee_mri --help` is equivalent to `knee-mri --help` after installation. The check and split commands and command help require only the standard library. Importing the package does not start training, load data, or download models.

Functions and methods use English docstrings. Business interfaces document arguments, return values, array shapes, and major exceptions; simple protocol and test methods use brief descriptions.
