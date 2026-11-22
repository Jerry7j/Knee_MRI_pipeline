"""Factories for PFDNet, ResNet34 U-Net, and MONAI 3D U-Net."""
from ..common import require_artifact, validate_architecture


def build_model(config, pretrained=False):
    """Create a five-class segmentation architecture without loading its checkpoint.

    Args:
        config: model_type (2.5d/3d), architecture (unet/pfdnet), and optional pfdnet_channels.
        pretrained: Load ImageNet weights for the 2.5D ResNet34 encoder when True.

    Returns:
        PFDNet, an SMP 2.5D U-Net, or a MONAI 3D UNet producing five-class logits.

    Raises:
        ValueError: The model type is unsupported.
        ImportError: Required architecture dependencies are unavailable.
    """
    architecture = validate_architecture({**config, "pretrained": pretrained or config.get("pretrained", False)})
    if architecture == "pfdnet":
        from .pfdnet import PFDNet
        return PFDNet(channel=config.get("pfdnet_channels", 128), classes=5)
    if config["model_type"] == "2.5d":
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=5)
    if config["model_type"] == "3d":
        from monai.networks.nets import UNet
        return UNet(spatial_dims=3, in_channels=1, out_channels=5, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
    raise ValueError(f"Unknown model type: {config['model_type']}")


def load_weights(model, path, device="cpu"):
    """Strictly load a model state dictionary or a training checkpoint.

    Args:
        model: Matching model architecture whose parameters are updated in place.
        path: Actual weight file, not a Git LFS pointer.
        device: map_location passed to torch.load.

    Returns:
        The loaded checkpoint or state dictionary.

    Raises:
        FileNotFoundError: The weight file is missing.
        ValueError: The file is a Git LFS pointer.
        RuntimeError: Weight keys, tensor shapes, or checkpoint format are incompatible.
    """
    import torch
    path = require_artifact(path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    return checkpoint
