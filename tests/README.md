# Tests

[English](README.md) | [简体中文](README.zh-CN.md)

```powershell
python -m unittest discover -s tests -v
# Alternatively, install the dev dependencies and run pytest
pytest
```

test_core.py uses only the standard library. It covers patient and repeat-visit leakage, invalid splits, determinism, volume study splits, configuration paths and dimensions, artifact checks, and command-line help.

test_scientific.py uses synthetic data to check neighbor fallback, nearest-neighbor mask resizing, alignment errors, ignored padding, Dice/IoU, resolution-group weights, probability validation, and entropy. It requires NumPy, PyTorch, torchvision, and Pillow; unavailable dependencies cause explicit skips.

Each test method has an English docstring describing its checks. Tests do not run full training or load actual MRI data or pretrained weights. Full MONAI, YOLO, DICOM, and application workflows require integration validation with the appropriate dependencies and data.
