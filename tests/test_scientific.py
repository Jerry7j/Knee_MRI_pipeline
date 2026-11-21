"""Synthetic tensor checks; skipped explicitly when core AI dependencies are absent."""
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
try:
    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    from knee_mri.data.slices import KneeSegmentation25D, pad_batch_to_32
    from knee_mri.engine.metrics import segmentation_scores
    from knee_mri.engine.fusion import align_probability, fuse_probabilities
    from knee_mri.visualization.plots import entropy_map, high_entropy_mask
    SCIENTIFIC_AVAILABLE, REASON = True, ""
except (ImportError, RuntimeError) as error:
    SCIENTIFIC_AVAILABLE, REASON = False, f"Scientific dependencies unavailable: {error}"


@unittest.skipUnless(SCIENTIFIC_AVAILABLE, REASON)
class TensorTests(unittest.TestCase):
    def test_padding_is_ignored_by_metrics(self):
        """Verify batch padding uses -100 labels and does not affect segmentation metrics."""
        batch = [(torch.ones(3, 19, 21), torch.ones(19, 21, dtype=torch.long)), (torch.ones(3, 32, 29), torch.ones(32, 29, dtype=torch.long))]
        images, masks = pad_batch_to_32(batch)
        self.assertEqual(tuple(images.shape), (2, 3, 32, 32))
        self.assertEqual(masks[0, 31, 31].item(), -100)
        pred = torch.ones_like(masks[0])
        pred[masks[0] == -100] = 4
        self.assertEqual(segmentation_scores(pred, masks[0])["iou"], 1.0)

    def test_missing_neighbors_repeat_center_and_masks_use_nearest(self):
        """Verify missing neighbors repeat the center and resizing preserves label classes."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "masks/001_visit").mkdir(parents=True)
            name = "001_visit_slice_001.jpg"
            Image.fromarray(np.full((12, 10), 128, np.uint8)).save(root / "images" / name)
            mask = np.zeros((12, 10), np.uint8)
            mask[3:8, 3:7] = 4
            np.save(root / "masks/001_visit/001_visit_slice_001.npy", mask)
            ds = KneeSegmentation25D(root / "images", root / "masks", [name], input_size=(32, 32))
            image, label = ds[0]
            self.assertTrue(torch.equal(image[0], image[1]) and torch.equal(image[1], image[2]))
            self.assertEqual(set(label.unique().tolist()), {0, 4})
            self.assertEqual(tuple(image.shape), (3, 32, 32))

    def test_foreground_macro_and_absent_class_policy(self):
        """Verify foreground macro metrics and the score-one policy for absent classes."""
        target = torch.tensor([[1, 1], [0, 0]])
        pred = torch.tensor([[1, 0], [0, 0]])
        scores = segmentation_scores(pred, target)
        self.assertAlmostEqual(scores["iou"], (0.5 + 3) / 4)
        self.assertAlmostEqual(scores["dice"], (2 / 3 + 3) / 4, places=6)

    def test_resolution_groups_have_equal_weight(self):
        """Verify resolution groups have equal weight regardless of their model counts."""
        low = torch.zeros(5, 8, 8)
        high = torch.zeros(5, 16, 16)
        low[1], high[2] = 1, 1
        fused = fuse_probabilities([low, low, low], [high], target_size=(16, 16))
        self.assertTrue(torch.allclose(fused[1], torch.full((16, 16), 0.5)))
        self.assertTrue(torch.allclose(fused[2], torch.full((16, 16), 0.5)))
        self.assertTrue(torch.allclose(fused.sum(0), torch.ones(16, 16)))

    def test_invalid_probabilities_rejected(self):
        """Verify invalid class probability sums and empty fusion groups are rejected."""
        with self.assertRaises(ValueError):
            align_probability(torch.zeros(5, 8, 8), (16, 16))
        with self.assertRaises(ValueError):
            fuse_probabilities([], [])

    def test_entropy_uniform_and_certain_predictions(self):
        """Verify entropy limits for uniform and certain predictions and percentile checks."""
        self.assertTrue(np.allclose(entropy_map(torch.full((5, 2, 2), 0.2)), np.log(5)))
        certain = torch.zeros(5, 2, 2)
        certain[1] = 1
        self.assertTrue(np.allclose(entropy_map(certain), 0))
        with self.assertRaises(ValueError):
            high_entropy_mask(np.zeros((2, 2)), 0)

    def test_mask_alignment_error_not_hidden_by_resize(self):
        """Verify resizing does not conceal image/mask alignment errors."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "masks/001_visit").mkdir(parents=True)
            name = "001_visit_slice_001.jpg"
            Image.fromarray(np.zeros((12, 10), np.uint8)).save(root / "images" / name)
            np.save(root / "masks/001_visit/001_visit_slice_001.npy", np.zeros((10, 12), np.uint8))
            ds = KneeSegmentation25D(root / "images", root / "masks", [name])
            with self.assertRaisesRegex(ValueError, "alignment"):
                ds[0]


if __name__ == "__main__":
    unittest.main()
