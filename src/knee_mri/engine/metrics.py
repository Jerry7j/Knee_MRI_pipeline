"""Foreground macro Dice/IoU; absent classes score one."""
def segmentation_scores(pred, target, num_classes=5, ignore_index=-100):
    """Compute foreground macro Dice/IoU for one sample.

    Args:
        pred: Predicted class labels shaped [H,W] or [D,H,W].
        target: Ground-truth labels with the same shape as pred.
        num_classes: Total class count, defaulting to five; background class zero
            is not scored separately.
        ignore_index: Target label excluded from calculations, defaulting to -100.

    Returns:
        A dictionary of dice, iou, dice_per_class, and iou_per_class.
        A foreground class absent from both arrays receives a score of one.

    Raises:
        ValueError: Shapes differ or no valid target pixels remain.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {target.shape}")
    valid = target != ignore_index
    if not valid.any():
        raise ValueError("No valid target pixels")
    dice, iou = [], []
    for cls in range(1, num_classes):
        p, t = (pred == cls) & valid, (target == cls) & valid
        intersection = (p & t).sum().item()
        denominator = p.sum().item() + t.sum().item()
        union = (p | t).sum().item()
        dice.append((2 * intersection + 1e-6) / (denominator + 1e-6))
        iou.append(intersection / union if union else 1.0)
    return {"dice": sum(dice) / len(dice), "iou": sum(iou) / len(iou), "dice_per_class": dice, "iou_per_class": iou}


def mean_scores(scores):
    """Compute equally weighted sample averages for five-class segmentation.

    Args:
        scores: Nonempty list of segmentation_scores results, each containing
            metrics for four foreground classes.

    Returns:
        Sample count, mean macro Dice/IoU, and four per-class averages.

    Raises:
        ValueError: The score list is empty.
    """
    if not scores:
        raise ValueError("No samples to evaluate")
    return {"samples": len(scores), **{key: sum(s[key] for s in scores) / len(scores) for key in ("dice", "iou")},
            **{key: [sum(s[key][c] for s in scores) / len(scores) for c in range(4)] for key in ("dice_per_class", "iou_per_class")}}
