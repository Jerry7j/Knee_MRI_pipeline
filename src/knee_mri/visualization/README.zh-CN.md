# 分割可视化

[English](README.md) | [简体中文](README.zh-CN.md)

`plots.py` 提供分割面板、熵图、高熵区域和训练曲线，使用 Agg 后端保存文件。

```powershell
knee-mri visualize --config configs/noise_512_k5_s015.json --exports outputs/fusion --output outputs/fusion/figures --limit 20 --mask-background
knee-mri curves --log outputs/noise_256_k5_s015/training_log.csv --output outputs/noise_256_k5_s015/curves.png
```

save_panel 接受二维中心图、[5,H,W] 概率和可选真值，保存中心图、真值、预测、分割叠加和熵叠加 PNG，并保存 entropy 与 high_entropy NPY。

颜色为背景黑、股骨绿、胫骨蓝、髌骨橙、半月板深红。熵 -sum(p*ln(p)) 单位为 nats，色标为 0 至 ln(5)。默认按最高 20% 分位数选择高熵区域，相同熵值可能使入选像素超过 20%。

--mask-background 按真值或预测前景控制叠加，完整熵值始终保存。批量 visualize 校验分辨率、resize 模式、split 和数据目录，支持 2.5D；3D 可选取切面后调用 save_panel。

plot_curves 读取 training_log.csv，保存训练/验证的 loss、Dice、IoU 曲线。
