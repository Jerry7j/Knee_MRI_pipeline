# 测试

[English](README.md) | [简体中文](README.zh-CN.md)

```powershell
python -m unittest discover -s tests -v
# Alternatively, install the dev dependencies and run pytest
pytest
```

test_core.py 仅依赖标准库，覆盖患者与重复访视泄漏、无效划分、确定性、3D study 划分、配置路径与尺寸、模型文件检查及命令行帮助。

test_scientific.py 用合成数据检查邻片回退、标签最近邻插值、对齐错误、padding 忽略、Dice/IoU、分辨率组权重、概率校验和熵。需要 NumPy、PyTorch、torchvision 和 Pillow，缺失时明确显示 skipped。

每个测试方法的英文文档字符串描述其验证行为。test_pfdnet.py 覆盖输出与梯度、配置校验、权重加载、一个 epoch 的合成数据训练、评估和多分辨率推理。测试不加载实际 MRI 或预训练权重；MONAI、YOLO、DICOM 与应用的完整流程需使用对应依赖和数据验证。
