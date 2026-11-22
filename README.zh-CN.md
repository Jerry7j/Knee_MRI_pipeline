# Knee MRI Pipeline

[English](README.md) | [简体中文](README.zh-CN.md)

用于膝关节 MRI 软骨与半月板分割的 Python 项目，支持数据预处理、2.5D/3D 模型训练、独立测试、多分辨率融合、可视化和交互推理。

## 合作背景

本项目用于与上海龙华医院合作开展膝关节 MRI 数据分析。研究使用上海龙华医院提供的膝关节 MRI 数据，重点分析软骨与半月板分割。

## 实现参考

本仓库的实现参考了 [pitthexai/Knee_MRI_Segmentation_2.5D](https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D) 开源项目，以及发表于 *npj Health Systems* 的论文 [KneeXNet-2.5D: a clinically-oriented and explainable deep learning framework for MRI-based knee cartilage and meniscus segmentation](https://www.nature.com/articles/s44401-026-00072-5)。感谢作者公开研究方法与代码。

## 流程概览

下图展示参考方法 KneeXNet-2.5D 的整体流程：标注与 YOLOv11 定位、多分辨率和高斯尺度的三邻片输入、U-Net 分割、集成预测及基于熵的不确定性可视化。

[![KneeXNet-2.5D 整体流程图](pipeline.png)](pipeline.png)

*图片来源：[pitthexai/Knee_MRI_Segmentation_2.5D — Figures/pipeline.png](https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D/blob/main/Figures/pipeline.png)。点击图片可查看完整尺寸。*

## 目录结构

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

四个核心模块分别是[数据](src/knee_mri/data/README.zh-CN.md)、[模型](src/knee_mri/models/README.zh-CN.md)、[运行流程](src/knee_mri/engine/README.zh-CN.md)和[可视化](src/knee_mri/visualization/README.zh-CN.md)。所有 Python 函数均有英文文档字符串，可通过编辑器悬停或 `help()` 查看。

## 安装

### 环境与依赖明细

| 环境项 | 要求 |
| --- | --- |
| Python | 项目支持 `>=3.10,<3.13`；uv 通过 `.python-version` 默认使用 **3.11** |
| uv | 锁文件使用 **0.13.9** 生成并校验，建议使用新版 uv |
| 默认后端 | 从 PyPI 安装 PyTorch **2.6.0**、torchvision **0.21.0** |
| NVIDIA 后端 | 可选本地安装：PyTorch **2.6.0+cu124**、torchvision **0.21.0+cu124**；需要 Windows/Linux x86_64 和兼容的 NVIDIA 驱动 |
| 构建依赖 | `setuptools>=68`、`wheel`，由构建工具按需安装 |

下表完整列出 `pyproject.toml` 中直接声明的核心和可选依赖。锁定版本来自 **Python 3.11 / Windows** 的安装计划；其他支持的 Python 版本或系统可能选择 `uv.lock` 中的不同版本条目。

| 功能组 | 依赖包 | 声明范围 | 锁定版本 | 用途 |
| --- | --- | --- | --- | --- |
| 核心 | numpy | `>=1.24,<3` | 2.4.6 | 切片、标签、概率数组和 NPY 数据 |
| 核心 | Pillow | `>=10` | 13.0.0 | JPEG/PNG 图像读写 |
| 核心 | torch | `==2.6.0` | 2.6.0 + 所选后端 | 网络训练、张量运算和推理 |
| 核心 | torchvision | `==0.21.0` | 0.21.0 + 所选后端 | 图像变换与模型工具 |
| 核心 | segmentation-models-pytorch | `>=0.4,<0.6` | 0.5.0 | 原有 ResNet34 U-Net |
| 核心 | matplotlib | `>=3.7` | 3.11.2 | 分割可视化和训练曲线 |
| 核心 | tqdm | `>=4.65` | 4.70.1 | 训练与评估进度显示 |
| `volume` | monai | `>=1.3,<2` | 1.5.2 | 3D U-Net |
| `volume` | nibabel | `>=5` | 5.5.0 | NIfTI 图像与标签读写 |
| `volume`、`localization` | SimpleITK | `>=2.3` | 2.5.8 | DICOM 序列与三维体数据读写 |
| `localization`、`app` | ultralytics | `>=8.3,<9` | 8.4.216 | YOLO ROI 定位 |
| `localization` | PyYAML | `>=6` | 6.0.4 | YOLO 数据集配置 |
| `app` | streamlit | `>=1.40,<2` | 1.67.0 | 交互推理界面 |
| `app` | pydicom | `>=3` | 3.0.2 | DICOM 元信息读取 |
| `app` | opencv-python | `>=4.8` | 5.1.0.95 | 图像处理和应用显示 |
| `dev` | pytest | `>=7` | 9.2.0 | 可选测试运行器；unittest 属于 Python 标准库 |

`uv.lock` 同时保存间接依赖及安装包哈希，包括 timm、huggingface-hub、safetensors、pandas、pyarrow，以及 Linux 下按后端选择的 NVIDIA 运行库，由 uv 随上层包自动安装。PFDNet 和 Res2Net 已包含在本仓库，直接使用 PyTorch，无需另装 PFDNet 包。

不使用界面、只推理 DICOM 序列时，也应选择 `localization` 以安装 SimpleITK；NIfTI/3D 处理选择 `volume`。模型 `.pt`/`.pth` 权重和 MRI 数据需单独准备，uv 不会下载。使用这些预编译 PyTorch 安装包无需额外安装系统 CUDA Toolkit；CPU 模式不需要 CUDA。CUDA 12.4 仅适用于兼容的 NVIDIA 环境；macOS Apple Silicon 使用 PyPI 安装包，本项目尚未验证该平台。

本次已校验锁文件，仅添加配置与说明，没有安装新环境；完整锁定依赖组合尚未执行应用测试。

### 使用 uv（推荐）

环境只由三个文件管理：`pyproject.toml` 声明依赖、`.python-version` 指定 Python 3.11、`uv.lock` 锁定解析后的版本，无需额外 uv 配置文件。需要安装时，在项目目录执行：

```powershell
python -m pip install --upgrade uv
uv python install 3.11
uv sync --locked --extra volume --extra localization --extra app --extra dev
uv run --no-sync knee-mri check
```

只需 2.5D 训练时，使用 `uv sync --locked`。运行命令时无需手动激活 `.venv`，前面加上 `uv run --no-sync` 即可：

```powershell
uv run --no-sync knee-mri train --config configs/pfdnet_25d.json
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync streamlit run apps/streamlit_app.py
```

默认从 PyPI 安装，是否支持 CUDA 取决于系统和安装的 PyTorch 构建版本。Windows/Linux x86_64 如需 CUDA 12.4 版，在同步环境**之后**按需执行：

```powershell
uv pip install --reinstall-package torch --reinstall-package torchvision torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv run --no-sync python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CUDA 替换属于本地选择，不由 `uv.lock` 锁定。后续使用 `uv run --no-sync` 保留该选择；再次执行 `uv sync` 后，如有需要重新运行 CUDA 安装命令。需要兼容的 NVIDIA 驱动，使用预编译安装包无需另外安装系统 CUDA Toolkit。

修改依赖后执行 `uv lock`，将更新后的锁文件与 `pyproject.toml` 一起提交，`.venv` 保留本地。本次仅创建配置和说明，不安装环境。

### 使用 pip

使用 Python 3.10 或 3.11，在本目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[volume,localization,app,dev]"
knee-mri --help
knee-mri check
```

只运行 2.5D 训练和测试可安装 `pip install -e .`。3D 需要 `volume`，YOLO 需要 `localization`，交互应用需要 `app`。`requirements.txt` 安装全部功能。GPU 版 PyTorch 应与 CUDA 环境匹配。

## 数据格式

2.5D 图像与标签按以下格式配对：

```text
data/processed/
+-- images/
|   +-- 9001104_20050825_10498205_slice_001.jpg
+-- masks/
    +-- 9001104_20050825_10498205/
        +-- 9001104_20050825_10498205_slice_001.npy
```

图像为灰度 JPEG，标签为相同尺寸的二维整数数组。三张相邻切片组成 `[3,H,W]` 输入，缺少邻片时复制中心切片。类别定义如下：

| ID | 类别 |
| --- | --- |
| 0 | 背景 |
| 1 | 股骨远端软骨 |
| 2 | 胫骨近端软骨 |
| 3 | 髌骨软骨 |
| 4 | 半月板 |

3D 数据使用 `data/processed/volumes/<study>/*.jpg` 和 `data/raw/annotations/<study>.nii.gz`。图像顺序、标签轴向和像素位置必须对齐；`mask_axis_order` 控制标签数组的轴排列。

患者 ID 为 study 名称第一个下划线前的部分。同一患者的不同访视归入同一集合，各分辨率实验共用相同划分。

## 训练与测试

先修改[实验配置](configs/README.zh-CN.md)中的数据路径。配置内相对路径按 `project_root` 解析，默认指向本项目根目录；命令行额外路径相对当前工作目录。

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

可选配置包括变尺寸 2.5D 基线、256/512 分辨率的四组模糊实验和 3D U-Net。3D 使用单独数据目录与划分：

```powershell
knee-mri split --config configs/unet3d.json
knee-mri train --config configs/unet3d.json
knee-mri evaluate --config configs/unet3d.json
```

训练输出 `best_model.pth`、`best.ckpt`、`last.ckpt`、CSV 日志和 JSON 摘要，按验证 IoU 选择最佳模型。重复使用输出目录会覆盖已有训练产物。

测试输出 `metrics.json`、逐样本 `manifest.json`、预测标签和 softmax 概率。`--checkpoint <path>` 指定权重，`--subset val` 选择验证集，`--no-probs` 关闭概率保存。Dice/IoU 对类别 1 至 4 宏平均，再按样本平均；预测和标签均为空的类别得分为 1。

## PFDNet 分割

加入作者的 [PFDNet 实现](src/knee_mri/models/pfdnet.py)，保留 Res2Net 编码器和两路并行融合解码结构，将输出层适配为背景、股骨软骨、胫骨软骨、髌骨软骨和半月板的 5 类分割 logits。输入仍为前、中、后三张 MRI 切片组成的 2.5D 数据。

训练配置使用 `"model_type": "2.5d"`、`"architecture": "pfdnet"` 和 `"pfdnet_channels": 32`。后者设置解码器宽度，训练与推理必须保持一致。未指定 architecture 的现有配置继续使用 U-Net。PFDNet 也支持原有不缩放、补齐到 32 倍数的数据流程。

```powershell
knee-mri split --config configs/pfdnet_25d.json
knee-mri train --config configs/pfdnet_25d.json
knee-mri evaluate --config configs/pfdnet_25d.json
knee-mri predict --config configs/ensemble_pfdnet.json --input data/raw/volume.npy --output outputs/pfdnet_inference
```

运行前请修改数据路径并训练 MRI 分割权重。现有关键点权重或已下载的 U-Net 权重不能直接加载到五类 PFDNet 分割模型。程序不会自动下载 PFDNet 预训练权重。YOLO 定位仍由原 localizer.pt 完成；集成配置的每个模型可分别选择 unet 或 pfdnet，并沿用原分辨率分组融合。

### PFDNet 论文引用

使用 PFDNet 架构时，请引用作者的论文，并注明论文与代码来源；保留相应版权和署名声明。

Wu Z, Zhang R, Bai H, Ma J, Ma X, Zhu X. [Keypoint estimation of knee x-ray images using a parallel fusion decoding network](https://www.sciencedirect.com/science/article/abs/pii/S0968016022002101). *The Knee*.

## 多分辨率融合

分别训练并测试四组 noise 配置，再执行：

```powershell
knee-mri fuse --config configs/noise_512_k5_s015.json --low outputs/noise_256_k5_s015/test outputs/noise_256_k7_s010/test --high outputs/noise_512_k5_s015/test outputs/noise_512_k7_s010/test --output outputs/fusion
knee-mri visualize --config configs/noise_512_k5_s015.json --exports outputs/fusion --output outputs/fusion/figures --mask-background
```

概率按目标尺寸双线性对齐，先组内平均，再对分辨率组等权平均。输入须为相同患者划分与裁剪版本的固定尺寸 2.5D 测试导出；融合会校验样本 ID、顺序和数据目录。

## YOLO 定位与裁剪

```powershell
knee-mri yolo-prepare --images data/raw/images --labels data/raw/yolo_labels --output data/processed/yolo
knee-mri yolo-train --data data/processed/yolo/KneeMRI.yaml --checkpoint yolo11n.pt --epochs 50 --output outputs/yolo
knee-mri yolo-evaluate --data data/processed/yolo/KneeMRI.yaml --checkpoint ../Models/localizer.pt --output outputs/yolo_test
knee-mri crop --images data/raw/images --annotations data/raw/annotations --checkpoint ../Models/localizer.pt --output data/processed --slice-start 1 --mask-axes 0 1 2
```

YOLO 使用同名 txt 标签，每行 `0 x_center y_center width height`，坐标归一化到 [0,1]；负样本提供空 txt。定位选择最高置信度框，并对图像与标签同步裁剪。数据准备和裁剪应使用新的输出目录。

裁剪标签通过 SimpleITK 读取，`--mask-axes` 针对其数组轴序；3D 数据集通过 nibabel 读取，两种读取方式的轴定义需分别核对。

## 无标签推理与应用

配置 [ensemble.json](configs/ensemble.json) 的模型路径、分辨率分组和可选定位器。默认权重路径指向 `../Models`，也可使用 `outputs/<experiment>/best_model.pth`。预先裁剪且对齐的输入可将 `localizer` 设为 null。

```powershell
knee-mri predict --config configs/ensemble.json --input data/raw/volume.npy --output outputs/inference
streamlit run apps/streamlit_app.py
```

NPY 输入为 `[N,H,W]`；DICOM 目录须包含同一序列的单帧灰度切片，按 InstanceNumber 排序。推理使用中央切片的定位框裁剪三张邻片，无检测时报告错误。

CLI 输出图像数组尺寸的 `segmentation.npy`、各切片概率和 ROI 坐标记录。应用可显示所选切片并下载 ROI 分割。NPY 输出不包含 NIfTI affine 或 DICOM 空间元数据。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试范围见 [tests/README.md](tests/README.zh-CN.md)。科学计算依赖缺失时，对应测试显示 skipped。完整训练与推理需要可用影像数据和真实模型权重；Git LFS 指针不能作为权重加载，可通过 `knee-mri check` 检查文件状态。
