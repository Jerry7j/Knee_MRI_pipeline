# 数据与预处理

[English](README.md) | [简体中文](README.zh-CN.md)

负责 MRI 输入、分割数据集、患者划分和 YOLO 定位裁剪。

| 文件 | 功能 |
| --- | --- |
| `splits.py` | 切片名解析、按患者分组、JSON 划分生成与校验 |
| `slices.py` | 三邻片 2.5D 数据集、缩放、模糊和补齐 |
| `volumes.py` | JPEG 体与 NIfTI 标签数据集、共享 3D 空间增强 |
| `loaders.py` | 按配置创建 train/val/test 数据加载器 |
| `inputs.py` | NPY 和单序列 DICOM 体读取 |
| `yolo.py` | 检测数据准备、YOLO 训练与验证/测试 |
| `crops.py` | 最高置信度框与图像/标签同步裁剪 |

## 分割数据

2.5D 使用平铺的 `<study>_slice_001.jpg` 和 `<mask_dir>/<study>/<slice_stem>.npy`，返回 image `[3,H,W]` float32 与 mask `[H,W]` int64。图像双线性缩放，标签最近邻缩放；缺少邻片时复制中心。变尺寸模式先中心补齐，再对 batch 右侧和下侧补齐至 32 倍数，额外标签为 -100。

3D 使用 `<image_dir>/<study>/*.jpg` 和 `<mask_dir>/<study>.nii.gz`，返回 `[1,D,H,W]` 与 `[D,H,W]`。mask_axis_order 配置 nibabel 标签数组的轴排列。训练图像与标签共享随机空间变换；验证/测试无随机增强。尺寸不匹配或标签不属于整数 0 至 4 时报告错误。

```powershell
knee-mri split --config configs/noise_256_k5_s015.json
```

患者 ID 为 study 的第一段。同患者的不同访视归入同一集合。train、val、test 均须非空，不允许重复样本或跨集合患者泄漏。

## 定位与裁剪

```powershell
knee-mri yolo-prepare --images data/raw/images --labels data/raw/yolo_labels --output data/processed/yolo
knee-mri yolo-train --data data/processed/yolo/KneeMRI.yaml --epochs 50 --output outputs/yolo
knee-mri yolo-evaluate --data data/processed/yolo/KneeMRI.yaml --checkpoint ../Models/localizer.pt
knee-mri crop --images data/raw/images --annotations data/raw/annotations --checkpoint ../Models/localizer.pt --output data/processed --slice-start 1 --mask-axes 0 1 2
```

YOLO txt 格式为 `0 x y w h`，坐标归一化，负样本须提供空 txt。准备结果包括 train/val/test 的 images、labels 和 KneeMRI.yaml。

裁剪使用 SimpleITK 读取标签，取最高置信度框同步裁剪，输出 images、masks 和 crops.json；无检测样本记录后跳过。两种操作均要求新的样本输出目录。

## 无标签输入

load_volume 接受 `[N,H,W]` NPY 或 .dcm 目录。DICOM 限同一序列的单帧灰度图，按 InstanceNumber 排序；重复序号或混合序列会报错。调用方须确认切片方向及标签坐标一致。
