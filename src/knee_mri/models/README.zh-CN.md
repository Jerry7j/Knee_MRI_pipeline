# 分割模型

[English](README.md) | [简体中文](README.zh-CN.md)

`factory.py` 提供 build_model 和 load_weights。

| model_type | 结构 | 输入与输出 |
| --- | --- | --- |
| `2.5d` | SMP U-Net，ResNet34 编码器，5 类 | 输入 [B,3,H,W]，输出 [B,5,H,W] logits |
| `2.5d`、`architecture=pfdnet` | PFDNet，Res2Net 编码器，共享并行解码，5 类 | 输入 [B,3,H,W]，输出 [B,5,H,W] logits |
| `3d` | MONAI UNet，通道 16/32/64/128/256，四级 stride=2，num_res_units=2 | 输入 [B,1,D,H,W]，输出 [B,5,D,H,W] logits |

2.5D 固定尺寸应为 32 的倍数，3D 各轴应为 16 的倍数。2.5D U-Net 训练可设置 pretrained=true 使用 ImageNet 编码器权重；PFDNet 必须设置 pretrained=false；测试和推理不下载编码器权重。

load_weights 接受 state_dict 或包含 model_state_dict 的 checkpoint，使用 weights_only 和 strict 加载。文件缺失、Git LFS 指针、模型键或形状不匹配时报告错误。

训练默认保存 `outputs/<experiment>/best_model.pth`。测试通过 --checkpoint 指定权重，集成推理通过 ensemble JSON 指定各模型。

## PFDNet

`pfdnet.py` 与 `res2net.py` 改编自作者的 `Keypoint-code/keypoint/models`。输入 [B,3,H,W]，输出与输入网格一致的五类原始 logits。两路解码共享聚合权重。设置 `architecture=pfdnet` 选择网络，`pfdnet_channels` 设置解码宽度（默认 128）。PFDNet 不接受 `pretrained=true`。论文引用：Wu Z, Zhang R, Bai H, Ma J, Ma X, Zhu X. [Keypoint estimation of knee x-ray images using a parallel fusion decoding network](https://www.sciencedirect.com/science/article/abs/pii/S0968016022002101). *The Knee*.
