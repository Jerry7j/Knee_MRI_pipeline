# 训练、测试与推理

[English](README.md) | [简体中文](README.zh-CN.md)

统一管理模型运行和概率后处理。

| 文件 | 接口与职责 |
| --- | --- |
| `trainer.py` | train、run_epoch：训练、验证与最佳模型保存 |
| `evaluator.py` | evaluate：独立 val/test 与逐样本导出 |
| `metrics.py` | segmentation_scores、mean_scores：前景 Dice/IoU |
| `predictor.py` | EnsemblePredictor、predict_volume：集成推理与 ROI 回贴 |
| `fusion.py` | align_probability、fuse_probabilities、fuse_exports：概率对齐和融合 |

## 训练与测试

```powershell
knee-mri train --config configs/noise_256_k5_s015.json
knee-mri evaluate --config configs/noise_256_k5_s015.json
```

训练使用 CrossEntropyLoss（忽略 -100）与 Adam，按验证 IoU 保存最佳模型。输出 best_model.pth、best.ckpt、last.ckpt、training_log.csv、config.json 和 summary.json。checkpoint 包含优化器与 epoch 信息，当前命令不提供断点续训。

测试使用 eval/inference 模式，输出预测、softmax 概率、manifest.json 与 metrics.json。2.5D 概率为 [5,H,W]，3D 为 [5,D,H,W]。--no-probs 关闭概率导出后不能继续融合或生成熵图。

Dice/IoU 对类别 1 至 4 宏平均，再按样本平均；背景不单独计分，类别两边均空时记为 1，-100 标签忽略。3D 每个体为一个样本。

## 融合

```powershell
knee-mri fuse --config configs/noise_512_k5_s015.json --low outputs/noise_256_k5_s015/test outputs/noise_256_k7_s010/test --high outputs/noise_512_k5_s015/test outputs/noise_512_k7_s010/test --output outputs/fusion
```

融合固定尺寸 2.5D 的 [5,H,W] softmax，校验概率非负、有限且类别和为 1。双线性对齐后组内平均，再对分辨率组等权平均。manifest 检查样本 ID、顺序、subset、图像与标签目录；输入应属于同一裁剪版本。

## 无标签推理

```powershell
knee-mri predict --config configs/ensemble.json --input data/raw/volume.npy --output outputs/inference
```

预测器按 ensemble JSON 加载模型组和可选 YOLO。三邻片共享中央框，边界复制中心；未检测到 ROI 时报告错误。融合概率插值回 ROI 尺寸后 argmax，再回贴到图像数组，ROI 外为背景。

输出 segmentation.npy [N,H,W]、逐切片模型尺寸概率和 prediction.json 坐标记录。此入口用于 2.5D 集成；3D 使用 evaluate 进行带标签测试。数组输出不包含物理空间元数据。
