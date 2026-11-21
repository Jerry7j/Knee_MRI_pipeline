"""Lazy command dispatch: --help and check work without AI dependencies."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from .common import load_config


def parser():
    """Build the command-line parser with all subcommands and arguments.

    Returns:
        An argparse.ArgumentParser without importing model dependencies or running tasks.
    """
    root = argparse.ArgumentParser(prog="knee-mri", description="Knee MRI modular research pipeline")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Report optional dependencies and model artifacts")
    split = commands.add_parser("split", help="Create patient-level train/val/test JSON")
    split.add_argument("--config", required=True)
    split.add_argument("--seed", type=int, default=42)
    for name in ("train", "evaluate"):
        p = commands.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "evaluate":
            p.add_argument("--checkpoint")
            p.add_argument("--output")
            p.add_argument("--subset", choices=["val", "test"], default="test")
            p.add_argument("--no-probs", action="store_true")
    fusion = commands.add_parser("fuse", help="Fuse probability exports and evaluate")
    fusion.add_argument("--config", required=True)
    fusion.add_argument("--low", nargs="*", default=[])
    fusion.add_argument("--high", nargs="*", default=[])
    fusion.add_argument("--output", required=True)
    vis = commands.add_parser("visualize")
    vis.add_argument("--config", required=True)
    vis.add_argument("--exports", required=True)
    vis.add_argument("--output", required=True)
    vis.add_argument("--limit", type=int, default=20)
    vis.add_argument("--mask-background", action="store_true")
    curves = commands.add_parser("curves")
    curves.add_argument("--log", required=True)
    curves.add_argument("--output", required=True)
    pred = commands.add_parser("predict", help="Predict unlabeled NPY/DICOM volumes with a 2.5D ensemble")
    pred.add_argument("--config", required=True)
    pred.add_argument("--input", required=True)
    pred.add_argument("--output", required=True)
    pred.add_argument("--device", default="auto")
    yp = commands.add_parser("yolo-prepare")
    yp.add_argument("--images", required=True)
    yp.add_argument("--labels", required=True)
    yp.add_argument("--output", required=True)
    yp.add_argument("--seed", type=int, default=42)
    yt = commands.add_parser("yolo-train")
    yt.add_argument("--data", required=True)
    yt.add_argument("--checkpoint", default="yolo11n.pt")
    yt.add_argument("--epochs", type=int, default=50)
    yt.add_argument("--output", default="outputs/yolo")
    yt.add_argument("--device")
    ye = commands.add_parser("yolo-evaluate")
    ye.add_argument("--data", required=True)
    ye.add_argument("--checkpoint", required=True)
    ye.add_argument("--subset", choices=["val", "test"], default="test")
    ye.add_argument("--output", default="outputs/yolo_eval")
    crop = commands.add_parser("crop")
    crop.add_argument("--images", required=True)
    crop.add_argument("--annotations", required=True)
    crop.add_argument("--checkpoint", required=True)
    crop.add_argument("--output", required=True)
    crop.add_argument("--slice-start", type=int, default=1)
    crop.add_argument("--mask-axes", nargs=3, type=int, default=[0, 1, 2])
    return root


def environment_report():
    """Inspect Python, optional dependencies, and artifacts in ../Models.

    Returns:
        A dictionary containing the Python version, dependency discovery results,
        model directory, and artifact presence or Git LFS pointer status.

    Notes:
        Does not import scientific libraries, load weights, or verify compatibility.
    """
    from .common import require_artifact
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "PIL", "torch", "torchvision", "segmentation_models_pytorch", "monai", "nibabel", "SimpleITK", "ultralytics", "streamlit", "pydicom", "matplotlib")}
    project = Path(__file__).resolve().parents[2]
    artifacts = {}
    model_dir = project.parent / "Models"
    for path in sorted(model_dir.glob("*")):
        if path.suffix not in (".pt", ".pth", ".pkl"):
            continue
        try:
            require_artifact(path)
            artifacts[path.name] = "present (not loaded or compatibility-tested)"
        except ValueError:
            artifacts[path.name] = "Git LFS pointer; actual model weights missing"
    return {"python_version": sys.version, "python_executable": sys.executable,
            "supported_python": sys.version_info >= (3, 10),
            "dependencies": dependencies, "artifacts": artifacts,
            "models_directory": str(model_dir), "models_directory_exists": model_dir.is_dir()}


def main(argv=None):
    """Parse arguments, dispatch the task, and print nonempty results as JSON.

    Args:
        argv: Argument list, or None to read the process command line.

    Raises:
        SystemExit: Argument parsing, configuration, file, or dependency checks fail.
    """
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, FileNotFoundError, ImportError) as error:
        raise SystemExit(str(error)) from error
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def dispatch(args):
    """Call the business entry point for a parsed subcommand.

    Args:
        args: An argparse.Namespace containing command and its task arguments.

    Returns:
        A task summary, metrics dictionary, or None. Business modules are imported
        only when their command is selected.
    """
    command = args.command
    if command == "check":
        return environment_report()
    if command == "split":
        from .data.splits import create_split
        config = load_config(args.config)
        result = create_split(config["image_dir"], config["split_path"], volume=config["model_type"] == "3d", seed=args.seed)
        return {key: len(value) for key, value in result.items()}
    if command == "train":
        from .engine.trainer import train
        return train(load_config(args.config))
    if command == "evaluate":
        from .engine.evaluator import evaluate
        return evaluate(load_config(args.config), args.checkpoint, args.output, args.subset, not args.no_probs)
    if command == "fuse":
        from .engine.fusion import fuse_exports
        return fuse_exports(load_config(args.config), args.low, args.high, args.output)
    if command == "visualize":
        from .visualization.plots import visualize_exports
        return visualize_exports(load_config(args.config), args.exports, args.output, args.limit, args.mask_background)
    if command == "curves":
        from .visualization.plots import plot_curves
        return plot_curves(args.log, args.output)
    if command == "predict":
        from .engine.predictor import predict_volume
        return predict_volume(args.config, args.input, args.output, args.device)
    if command == "yolo-prepare":
        from .data.yolo import prepare_yolo_dataset
        return {"data_yaml": str(prepare_yolo_dataset(args.images, args.labels, args.output, args.seed))}
    if command == "yolo-train":
        from .data.yolo import train_yolo
        train_yolo(args.data, args.checkpoint, args.epochs, args.output, args.device)
        return {"output": args.output}
    if command == "yolo-evaluate":
        from .data.yolo import evaluate_yolo
        return evaluate_yolo(args.checkpoint, args.data, args.subset, args.output)
    if command == "crop":
        from .data.crops import crop_dataset
        return crop_dataset(args.images, args.annotations, args.checkpoint, args.output, args.slice_start, tuple(args.mask_axes))


if __name__ == "__main__":
    main()
