# Segmentation models

[English](README.md) | [简体中文](README.zh-CN.md)

`factory.py` provides build_model and load_weights.

| model_type | Architecture | Input and output |
| --- | --- | --- |
| `2.5d` | SMP U-Net, ResNet34 encoder, 5 classes | Input [B,3,H,W], output [B,5,H,W] logits |
| `3d` | MONAI UNet, channels 16/32/64/128/256, four stride=2 levels, num_res_units=2 | Input [B,1,D,H,W], output [B,5,D,H,W] logits |

Fixed 2.5D dimensions must be multiples of 32; each 3D dimension must be a multiple of 16. Set pretrained=true to use ImageNet encoder weights for 2.5D training. Evaluation and inference do not download encoder weights.

load_weights accepts a state_dict or a checkpoint containing model_state_dict, using weights_only and strict loading. Missing files, Git LFS pointers, incompatible keys, or shape mismatches raise errors.

Training saves `outputs/<experiment>/best_model.pth` by default. Evaluation selects weights with --checkpoint; ensemble inference specifies each model in ensemble JSON.
