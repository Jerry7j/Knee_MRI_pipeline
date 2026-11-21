# Training, evaluation, and inference

[English](README.md) | [简体中文](README.zh-CN.md)

Model execution and probability postprocessing.

| File | Interfaces and responsibilities |
| --- | --- |
| `trainer.py` | train, run_epoch: training, validation, and best-model saving |
| `evaluator.py` | evaluate: independent val/test and per-sample exports |
| `metrics.py` | segmentation_scores, mean_scores: foreground Dice/IoU |
| `predictor.py` | EnsemblePredictor, predict_volume: ensemble inference and ROI restoration |
| `fusion.py` | align_probability, fuse_probabilities, fuse_exports: probability alignment and fusion |

## Training and evaluation

```powershell
knee-mri train --config configs/noise_256_k5_s015.json
knee-mri evaluate --config configs/noise_256_k5_s015.json
```

Training uses CrossEntropyLoss (ignoring -100) and Adam, saving the best model by validation IoU. Outputs include best_model.pth, best.ckpt, last.ckpt, training_log.csv, config.json, and summary.json. Checkpoints contain optimizer and epoch information; the current command does not support resuming training.

Evaluation uses eval/inference mode and exports predictions, softmax probabilities, manifest.json, and metrics.json. Probability shapes are [5,H,W] for 2.5D and [5,D,H,W] for 3D. Disabling probability exports with --no-probs prevents later fusion and entropy visualization.

Dice/IoU are macro-averaged over classes 1 through 4, then averaged across samples. Background is not scored separately; a class absent from both arrays scores one, and -100 labels are ignored. Each 3D volume is one sample.

## Fusion

```powershell
knee-mri fuse --config configs/noise_512_k5_s015.json --low outputs/noise_256_k5_s015/test outputs/noise_256_k7_s010/test --high outputs/noise_512_k5_s015/test outputs/noise_512_k7_s010/test --output outputs/fusion
```

Fusion accepts fixed-size 2.5D softmax maps [5,H,W], validating finite, nonnegative probabilities whose class sums equal one. Bilinear alignment is followed by within-group averaging and equal averaging across resolution groups. Manifest checks cover sample IDs, order, subset, and image/mask directories. Inputs must use the same crop version.

## Unlabeled inference

```powershell
knee-mri predict --config configs/ensemble.json --input data/raw/volume.npy --output outputs/inference
```

The predictor loads model groups and an optional YOLO localizer from ensemble JSON. Three neighboring slices share the central box; edge slices repeat the center. Missing ROIs raise errors. Fused probabilities are resized to ROI dimensions before argmax, then restored to the image grid with background outside the ROI.

Outputs include segmentation.npy [N,H,W], per-slice probabilities at model resolution, and prediction.json coordinate records. This entry point supports 2.5D ensembles; use evaluate for labeled 3D data. Array outputs contain no physical spatial metadata.
