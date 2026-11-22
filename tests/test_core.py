"""Dependency-free checks for leakage prevention, config paths, artifacts, and CLI."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from knee_mri.cli import main, parser
from knee_mri.data.splits import create_split, load_split, parse_slice, validate_split
from knee_mri.common import load_config
from knee_mri.common import require_artifact, write_json


class SplitTests(unittest.TestCase):
    def test_parse_study_with_multiple_underscores(self):
        """Verify study names with multiple underscores retain the correct slice index."""
        self.assertEqual(parse_slice("9001104_20050825_10498205_slice_001.jpg"), ("9001104_20050825_10498205", 1))

    def test_malformed_slice_rejected(self):
        """Verify malformed slice filenames are rejected."""
        with self.assertRaises(ValueError):
            parse_slice("not-a-slice.jpg")

    def test_patient_repeat_visits_cannot_cross_subsets(self):
        """Verify repeated visits from one patient cannot cross dataset subsets."""
        split = {"train": ["001_visit1_slice_001.jpg"], "val": ["001_visit2_slice_020.jpg"], "test": ["002_visit1_slice_001.jpg"]}
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_split(split)

    def test_empty_duplicate_and_path_entries_rejected(self):
        """Verify empty subsets, duplicate samples, and path-containing entries are rejected."""
        good = {"train": ["001_v_slice_001.jpg"], "val": ["002_v_slice_001.jpg"], "test": ["003_v_slice_001.jpg"]}
        for bad in ([], good["train"] * 2, ["../001_v_slice_001.jpg"], ["x\\001_v_slice_001.jpg"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_split({**good, "train": bad})

    def test_volume_repeat_visits_cannot_cross_subsets(self):
        """Verify volume splits keep repeated visits from one patient together."""
        with self.assertRaises(ValueError):
            validate_split({"train": ["001_visit1"], "val": ["001_visit2"], "test": ["002_visit1"]}, volume=True)

    def test_deterministic_split_groups_all_visits(self):
        """Verify seeded splits are reproducible and keep all patient visits together."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for patient in range(20):
                for visit in (1, 2):
                    for index in (1, 2, 3):
                        (root / f"{patient:03d}_{visit}_slice_{index:03d}.jpg").touch()
            first = create_split(root, root / "a.json", seed=42)
            second = create_split(root, root / "b.json", seed=42)
            self.assertEqual(first, second)
            self.assertEqual(sum(map(len, first.values())), 120)
            self.assertEqual([len(first[k]) for k in ("train", "val", "test")], [96, 12, 12])
            self.assertEqual(load_split(root / "a.json"), first)

    def test_insufficient_patients_and_bad_fractions(self):
        """Verify insufficient patients and invalid split fractions raise errors."""
        with tempfile.TemporaryDirectory() as tmp:
            for kwargs in ({}, {"val_fraction": 0.8, "test_fraction": 0.8}):
                with self.assertRaises(ValueError):
                    create_split(tmp, Path(tmp) / "split.json", **kwargs)

    def test_volume_split(self):
        """Verify study directories produce a complete volume split without patient leakage."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for patient in range(6):
                (root / f"{patient:03d}_scan").mkdir()
            split = create_split(root, root / "split.json", volume=True)
            self.assertEqual(sum(map(len, split.values())), 6)
            validate_split(split, volume=True)


class ConfigArtifactTests(unittest.TestCase):
    def test_all_experiment_configs_load(self):
        """Verify every training configuration loads with outputs inside the project."""
        paths = [p for p in (PROJECT / "configs").glob("*.json") if not p.stem.startswith("ensemble")]
        self.assertEqual({p.stem for p in paths}, {"baseline_25d", "noise_256_k5_s015", "noise_256_k7_s010",
                                                  "noise_512_k5_s015", "noise_512_k7_s010", "unet3d", "pfdnet_25d"})
        for path in paths:
            config = load_config(path)
            self.assertTrue(Path(config["image_dir"]).is_absolute())
            self.assertEqual(Path(config["output_dir"]).relative_to(PROJECT).parts[0], "outputs")

    def test_paths_resolve_against_config_root(self):
        """Verify relative data paths resolve against the configured project root."""
        config = load_config(PROJECT / "configs/noise_256_k5_s015.json")
        self.assertEqual(Path(config["image_dir"]), PROJECT / "data/processed/images")

    def test_bad_shapes_and_model_rejected(self):
        """Verify invalid model types, dimensions, and epoch counts are rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            for changes in ({"input_size": [255, 256]}, {"input_size": [0, 256]}, {"model_type": "unknown"}, {"epochs": 0}):
                path = Path(tmp) / "config.json"
                write_json(path, {"model_type": "2.5d", "input_size": [256, 256], **changes})
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_lfs_pointer_rejected_and_binary_accepted(self):
        """Verify artifact checks reject LFS pointers but accept ordinary binary files."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weight.pth"
            path.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n")
            with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
                require_artifact(path)
            path.write_bytes(b"PK\x03\x04binary-placeholder")
            self.assertEqual(require_artifact(path), path)

    def test_missing_artifact_rejected(self):
        """Verify missing model artifacts raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            require_artifact(PROJECT / "does-not-exist.pth")


class InterfaceTests(unittest.TestCase):
    def test_each_cli_command_has_help(self):
        """Verify every subcommand displays help without starting a business task."""
        choices = parser()._subparsers._group_actions[0].choices
        self.assertEqual(len(choices), 12)
        for command in choices:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                main([command, "--help"])
            self.assertEqual(result.exception.code, 0)

if __name__ == "__main__":
    unittest.main()
