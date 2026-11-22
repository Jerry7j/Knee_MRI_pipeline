# 实验配置

[English](README.md) | [简体中文](README.zh-CN.md)

| 文件 | 设置 |
| --- | --- |
| `baseline_25d.json` | 变尺寸 2.5D，三邻片，中心与 batch 补齐，无模糊 |
| `noise_256_k5_s015.json` | 256×256，GaussianBlur kernel=5、sigma=0.15 |
| `noise_256_k7_s010.json` | 256×256，kernel=7、sigma=0.10 |
| `noise_512_k5_s015.json` | 512×512，kernel=5、sigma=0.15 |
| `noise_512_k7_s010.json` | 512×512，kernel=7、sigma=0.10 |
| `unet3d.json` | MONAI 3D U-Net，112×224×224，2 epoch |
| `ensemble.json` | 分割模型列表、分辨率分组与可选定位器 |

训练配置通过 image_dir、mask_dir、split_path、output_dir 指定数据和输出路径，相对路径按 project_root 解析。四组 noise 配置共用 patient_split.json，默认 30 epoch、batch=8、Adam lr=1e-4、seed=65。

device=auto 自动选择 CPU/CUDA；Windows 默认 num_workers=0。pretrained=false 禁用 ImageNet 权重下载，true 可用于 2.5D 编码器预训练。resize=false 时 input_size 不用于缩放。

3D 的 mask_axis_order 是 nibabel 标签数组到 [D,H,W] 的轴排列，需依据数据设置。

ensemble 每个模型指定 checkpoint、size 和 group（low/high）。概率组内平均后组间等权融合。checkpoint 和 localizer 默认指向 ../Models，可改为训练输出路径；localizer=null 用于已裁剪且对齐的体。训练配置与集成配置不能互换。

`pfdnet_25d.json` 通过 `architecture=pfdnet`、`pfdnet_channels=32` 选择五类 PFDNet。`ensemble_pfdnet.json` 加载其训练权重。ensemble 每个模型可以设置 `architecture`（默认 `unet`）和 `pfdnet_channels`。PFDNet 支持 2.5D，示例不使用预训练权重。
