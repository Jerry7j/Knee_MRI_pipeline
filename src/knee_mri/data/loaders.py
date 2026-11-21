"""Construct split-specific loaders; no augmentation during evaluation."""
from torch.utils.data import DataLoader
from .splits import load_split
from .slices import KneeSegmentation25D, pad_batch_to_32


def make_dataset(config, subset, training=False):
    """Build a segmentation dataset for a model type and fixed split.

    Args:
        config: Experiment configuration with resolved paths.
        subset: train, val, or test.
        training: Enable configured image blur or shared 3D spatial augmentation.

    Returns:
        A KneeSegmentation25D or KneeVolumeDataset3D instance.
    """
    volume = config["model_type"] == "3d"
    samples = load_split(config["split_path"], volume)[subset]
    common = dict(image_dir=config["image_dir"], mask_dir=config["mask_dir"], input_size=config["input_size"])
    if volume:
        from .volumes import KneeVolumeDataset3D
        return KneeVolumeDataset3D(study_ids=samples, training=training, mask_axis_order=config.get("mask_axis_order", [0, 1, 2]), **common)
    return KneeSegmentation25D(filenames=samples, resize=config.get("resize", True),
                               blur_kernel=config.get("blur_kernel", 0) if training else 0,
                               blur_sigma=config.get("blur_sigma", 0.15), **common)


def make_loader(config, subset, training=False):
    """Create a data loader for the requested subset.

    Args:
        config: Configuration containing batch_size, num_workers, and data paths.
        subset: train, val, or test.
        training: Enable shuffling and training augmentation when True.

    Returns:
        A DataLoader that retains the final batch. Variable-size 2.5D samples use
        a dedicated padding collate function.
    """
    dataset = make_dataset(config, subset, training)
    collate = pad_batch_to_32 if config["model_type"] == "2.5d" and not config.get("resize", True) else None
    return DataLoader(dataset, batch_size=config.get("batch_size", 8), shuffle=training,
                      num_workers=config.get("num_workers", 0), drop_last=False, collate_fn=collate)
