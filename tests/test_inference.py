"""No downloaded weights, dataset access, or pretrained model execution."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import uuid
from contextlib import contextmanager
import unittest
from unittest.mock import patch

from scripts import infer_panoptic as inference


@contextmanager
def temporary_directory():
    # Keep fixtures within the writable results mount, including restricted hosts.
    root = inference.project_path("results")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"inference-smoke-{uuid.uuid4().hex}"
    # Inherited Windows permissions avoid owner-only TemporaryDirectory ACLs
    # that can exclude the executing account.
    path.mkdir()
    try:
        yield str(path)
    finally:
        if path.resolve().parent != root.resolve():
            raise RuntimeError("Refusing cleanup outside the test results directory")
        shutil.rmtree(path)


class PathAndErrorTests(unittest.TestCase):
    def test_missing_checkpoint_explains_offline_source(self):
        with temporary_directory() as directory:
            path = Path(directory) / "missing.bin"
            with self.assertRaisesRegex(FileNotFoundError, "No download was attempted"):
                inference.require_checkpoint(path)

    def test_cli_missing_checkpoint_exits_before_model_imports(self):
        with temporary_directory() as directory:
            with patch.object(inference, "load_model", side_effect=AssertionError("must not load")):
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    status = inference.main(["--input", "missing.png", "--checkpoint", str(Path(directory) / "missing.bin")])
            self.assertEqual(status, 2)
            self.assertIn(inference.MODEL_ID, stderr.getvalue())

    def test_cli_help_without_ml_dependencies(self):
        result = subprocess.run([sys.executable, str(inference.ROOT / "scripts/infer_panoptic.py"), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)

    def test_compose_relative_paths(self):
        with patch.dict(os.environ, {"EOMT_RESULTS_DIR": "/results"}):
            self.assertEqual(inference.project_path("results/run"), Path("/results/run").resolve())
        self.assertEqual(inference.project_path("upstream/eomt"), inference.UPSTREAM)

    def test_image_collection_and_empty_directory(self):
        with temporary_directory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "No supported images"):
                inference.collect_images([directory])
            (root / "image.PNG").write_bytes(b"placeholder")
            (root / "ignore.txt").write_text("not an image")
            self.assertEqual(inference.collect_images([directory, str(root / "image.PNG")]), [root / "image.PNG"])

    def test_output_outside_results_rejected_before_loading(self):
        with temporary_directory() as directory:
            checkpoint = Path(directory) / "local.bin"
            checkpoint.write_bytes(b"not loaded")
            image = Path(directory) / "input.png"
            image.write_bytes(b"not decoded")
            with patch.object(inference, "load_model", side_effect=AssertionError("must not load")):
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    status = inference.main(["--checkpoint", str(checkpoint), "--input", str(image),
                                             "--output-dir", str(inference.ROOT / "outside-results")])
            self.assertEqual(status, 2)
            self.assertIn("underneath", stderr.getvalue())


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL"), "requires NumPy and Pillow")
class OutputTests(unittest.TestCase):
    def test_outputs_preserve_ids_void_and_borders(self):
        import numpy as np
        from PIL import Image

        image = Image.new("RGB", (8, 6), (100, 150, 200))
        pred = np.zeros((6, 8, 2), dtype=np.int32)
        pred[:, 4:, 0], pred[:, 4:, 1] = 80, 1
        pred[0, 0] = [133, -1]
        metadata = {"test_only": True, "category_mapping": {0: 1, 80: 92}}
        with temporary_directory() as directory:
            destination = Path(directory) / "prediction"
            inference.save_prediction(image, pred, destination, metadata)
            with np.load(destination / "prediction.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["semantic"], pred[..., 0])
                np.testing.assert_array_equal(saved["segment_ids"], pred[..., 1])
            record = json.loads((destination / "prediction.json").read_text())
            self.assertEqual(record["void_segment_id"], -1)
            self.assertEqual(len(record["segments_info"]), 2)
            self.assertEqual(record["segments_info"][1]["coco_category_id"], 92)
            self.assertEqual(record["segments_info"][0]["area"], 23)
            self.assertFalse(record["segments_info"][1]["isthing"])
            with Image.open(destination / "segmentation.png") as colored:
                self.assertEqual(colored.size, image.size)
                self.assertEqual(colored.getpixel((0, 0)), (0, 0, 0))
                self.assertEqual(colored.getpixel((3, 3)), (0, 0, 0))
            self.assertTrue((destination / "overlay.png").is_file())
            with self.assertRaises(FileExistsError):
                inference.save_prediction(image, pred, destination, {})


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires the Docker PyTorch environment")
class UpstreamIntegrationTests(unittest.TestCase):
    def test_incompatible_checkpoint_is_not_silently_accepted(self):
        inference.offline_environment()
        import torch
        from training.mask_classification_panoptic import MaskClassificationPanoptic

        with patch("torch.load", return_value={"unrelated.weight": torch.zeros(1)}):
            with self.assertRaisesRegex(ValueError, "Missing keys"):
                MaskClassificationPanoptic(
                    network=torch.nn.Linear(1, 1), img_size=(640, 640), num_classes=133,
                    stuff_classes=list(range(80, 133)), attn_mask_annealing_enabled=False,
                    ckpt_path="constructor-double.bin",
                )

    def test_synthetic_logits_through_official_helpers(self):
        """Exercise resize/pad, restored sizes, stuff merging and output export."""
        inference.offline_environment()
        import numpy as np
        import torch
        from PIL import Image
        from training.lightning_module import LightningModule

        class SyntheticModel(torch.nn.Module):
            img_size = (32, 32)
            num_classes = 133
            stuff_classes = list(range(80, 133))
            mask_thresh = overlap_thresh = 0.8
            scale_img_size_instance_panoptic = LightningModule.scale_img_size_instance_panoptic
            resize_and_pad_imgs_instance_panoptic = LightningModule.resize_and_pad_imgs_instance_panoptic
            revert_resize_and_pad_logits_instance_panoptic = LightningModule.revert_resize_and_pad_logits_instance_panoptic
            to_per_pixel_preds_panoptic = LightningModule.to_per_pixel_preds_panoptic

            def forward(self, images):
                self.captured = images
                masks = torch.full((1, 3, 32, 32), -12.0)
                masks[:, 0, :, :10] = 12
                masks[:, 1, :, 10:21] = 12
                masks[:, 2, :, 21:] = 12
                classes = torch.full((1, 3, 134), -12.0)
                classes[0, 0, 80] = classes[0, 1, 80] = classes[0, 2, 0] = 12
                return [masks], [classes]

        model = SyntheticModel().eval()
        image = Image.new("RGB", (60, 30), (255, 100, 50))
        # Socket attempts fail the test even if a library regresses to downloading.
        with patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
            prediction = inference.predict_image(model, image)
        self.assertEqual(prediction.shape, (30, 60, 2))
        self.assertEqual(model.captured.dtype, torch.uint8)
        self.assertEqual(tuple(model.captured.shape), (1, 3, 32, 32))
        self.assertEqual(int(model.captured[:, :, 16:].sum()), 0)
        self.assertEqual(set(np.unique(prediction[..., 0])), {0, 80})
        self.assertEqual(len(np.unique(prediction[..., 1][prediction[..., 0] == 80])), 1)
        with temporary_directory() as directory:
            inference.save_prediction(image, prediction, Path(directory) / "synthetic", {"test_only": True})

    def test_local_loader_configuration_without_building_a_backbone(self):
        """Verify download suppression/config wiring using constructor doubles."""
        inference.offline_environment()
        import torch
        from models.vit import ViT

        # Official ViT must request pretrained=False for a supplied local file.
        fake_backbone = torch.nn.Identity()
        with patch("models.vit.timm.create_model", return_value=fake_backbone) as create:
            ViT(img_size=(640, 640), backbone_name="vit_small_patch14_reg4_dinov2", ckpt_path="local.bin")
        self.assertFalse(create.call_args.kwargs["pretrained"])
        self.assertEqual(create.call_args.kwargs["patch_size"], 16)

        with temporary_directory() as directory:
            checkpoint = Path(directory) / "stub.bin"
            checkpoint.write_bytes(b"constructor double only; never deserialized")
            with patch("models.vit.ViT") as encoder, patch("models.eomt.EoMT") as network, \
                    patch("training.mask_classification_panoptic.MaskClassificationPanoptic") as wrapper, \
                    patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
                inference.load_model(checkpoint)
            self.assertEqual(encoder.call_args.kwargs["ckpt_path"], str(checkpoint))
            self.assertEqual(network.call_args.kwargs["num_q"], 200)
            self.assertEqual(network.call_args.kwargs["num_blocks"], 3)
            self.assertFalse(network.call_args.kwargs["masked_attn_enabled"])
            self.assertEqual(wrapper.call_args.kwargs["num_classes"], 133)
            self.assertEqual(wrapper.call_args.kwargs["stuff_classes"], list(range(80, 133)))
            self.assertEqual(wrapper.call_args.kwargs["ckpt_path"], str(checkpoint))
            self.assertFalse(wrapper.call_args.kwargs["delta_weights"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
