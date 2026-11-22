"""PFDNet segmentation training, checkpoint, and inference integration checks."""

import contextlib
import gc
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

try:
    import numpy as np
    from PIL import Image
    import torch
    from knee_mri.models.factory import build_model, load_weights
    from knee_mri.models.pfdnet import PFDNet
    from knee_mri.engine.trainer import train
    from knee_mri.engine.evaluator import evaluate
    from knee_mri.engine.predictor import EnsemblePredictor, predict_volume
    AVAILABLE, REASON = True, ""
except ImportError as error:
    AVAILABLE, REASON = False, str(error)


@unittest.skipUnless(AVAILABLE, REASON)
class PFDNetTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)
        gc.collect()

    def test_segmentation_geometry_and_gradient(self):
        """Verify non-square input geometry and finite encoder/decoder gradients."""
        model = build_model({"model_type": "2.5d", "architecture": "pfdnet", "pfdnet_channels": 8})
        self.assertIsInstance(model, PFDNet)
        image = torch.rand(2, 3, 65, 81)
        labels = torch.randint(0, 5, (2, 65, 81))
        logits = model(image)
        self.assertEqual(logits.shape, (2, 5, 65, 81))
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        for parameter in (model.resnet.conv1[0].weight, model.agg1.conv5.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_settings_and_default_unet(self):
        """Reject incompatible PFDNet settings and preserve the default U-Net."""
        from knee_mri.common import validate_architecture
        for options in ({"model_type": "3d"}, {"pfdnet_channels": 0},
                        {"pfdnet_channels": True}, {"pretrained": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_architecture({"model_type": "2.5d", "architecture": "pfdnet", **options})
        with self.assertRaisesRegex(ValueError, "pretrained"):
            build_model({"model_type": "2.5d", "architecture": "pfdnet"}, pretrained=True)
        try:
            import segmentation_models_pytorch
        except ImportError:
            self.skipTest("U-Net dependency not installed")
        default = build_model({"model_type": "2.5d"})
        self.assertIsInstance(default, segmentation_models_pytorch.Unet)

    def test_train_evaluate_and_multiresolution_predict(self):
        """Train and evaluate synthetic labels, reload weights, and fuse two resolutions."""
        from knee_mri.common import load_config, write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, masks = root / "images", root / "masks"
            images.mkdir()
            split = {"train": [], "val": [], "test": []}
            for patient, subset in (("001", "train"), ("002", "train"), ("003", "val"), ("004", "test")):
                study = patient + "_visit"
                name = study + "_slice_001.jpg"
                pixels = (np.indices((64, 64)).sum(0) * 2).astype(np.uint8)
                Image.fromarray(pixels).save(images / name)
                (masks / study).mkdir(parents=True)
                target = np.zeros((64, 64), dtype=np.uint8)
                for label in range(1, 5):
                    target[8 * label:8 * label + 6, 10:40] = label
                np.save(masks / study / (Path(name).stem + ".npy"), target, allow_pickle=False)
                split[subset].append(name)
            write_json(root / "split.json", split)
            options = {"project_root": ".", "model_type": "2.5d", "architecture": "pfdnet",
                       "pfdnet_channels": 8, "image_dir": "images", "mask_dir": "masks",
                       "split_path": "split.json", "output_dir": "training", "input_size": [64, 64],
                       "resize": True, "batch_size": 2, "epochs": 1, "num_workers": 0, "device": "cpu",
                       "pretrained": False, "blur_kernel": 0}
            write_json(root / "experiment.json", options)
            config = load_config(root / "experiment.json")
            with contextlib.redirect_stdout(io.StringIO()):
                summary = train(config)
                metrics = evaluate(config)
            self.assertTrue(Path(summary["checkpoint"]).is_file())
            self.assertTrue((root / "training/test/metrics.json").is_file())
            saved = torch.load(root / "training/best.ckpt", map_location="cpu", weights_only=True)
            self.assertEqual(saved["config"]["architecture"], "pfdnet")
            del saved
            full_model = build_model(config).eval()
            load_weights(full_model, root / "training/best.ckpt")
            with torch.inference_mode():
                self.assertEqual(full_model(torch.rand(1, 3, 64, 64)).shape, (1, 5, 64, 64))
            del full_model
            specs = [{"architecture": "pfdnet", "pfdnet_channels": 8, "size": size,
                      "group": group, "checkpoint": "training/best_model.pth"}
                     for size, group in ((64, "low"), (96, "high"))]
            write_json(root / "ensemble.json", {"project_root": ".", "target_size": [64, 64],
                                              "localizer": None, "models": specs})
            volume = np.random.default_rng(7).random((2, 65, 81), dtype=np.float32)
            np.save(root / "input.npy", volume, allow_pickle=False)
            result = predict_volume(root / "ensemble.json", root / "input.npy", root / "prediction", "cpu")
            self.assertEqual(np.load(result["segmentation"]).shape, volume.shape)
            records = json.loads((root / "prediction/prediction.json").read_text())["samples"]
            for row in records:
                probability = np.load(root / "prediction" / row["probability"])
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue(np.allclose(probability.sum(0), 1, atol=1e-5))
            self.assertEqual(result["slices"], 2)


if __name__ == "__main__":
    unittest.main()
