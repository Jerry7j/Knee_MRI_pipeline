# Segmentation visualization

[English](README.md) | [简体中文](README.zh-CN.md)

`plots.py` provides segmentation panels, entropy maps, high-entropy regions, and learning curves, saving files with the Agg backend.

```powershell
knee-mri visualize --config configs/noise_512_k5_s015.json --exports outputs/fusion --output outputs/fusion/figures --limit 20 --mask-background
knee-mri curves --log outputs/noise_256_k5_s015/training_log.csv --output outputs/noise_256_k5_s015/curves.png
```

save_panel accepts a two-dimensional central image, [5,H,W] probabilities, and optional ground truth. It saves a PNG panel with the image, ground truth, prediction, segmentation overlay, and entropy overlay, plus entropy and high_entropy NPY arrays.

Colors are black for background, green for femoral cartilage, blue for tibial cartilage, orange for patellar cartilage, and dark red for meniscus. Entropy -sum(p*ln(p)) is measured in nats, with a color scale from zero to ln(5). High-entropy regions use the top 20% threshold by default; tied values can select more than 20% of pixels.

--mask-background restricts the overlay to ground-truth or predicted foreground; complete entropy values are always saved. Batch visualization validates resolution, resize mode, split, and data directories and supports 2.5D. For 3D, select a plane and call save_panel.

plot_curves reads training_log.csv and saves training/validation loss, Dice, and IoU curves.
