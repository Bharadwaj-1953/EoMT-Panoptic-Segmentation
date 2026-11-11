"""Synthetic only: real upstream forward on tiny random weights; no COCO/weights."""
import copy
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import evaluate_ablations as runner
from scripts import infer_panoptic as inference
from scripts.ablation_interventions import Intervention, apply_intervention, select_ids
from tests.test_inference import temporary_directory
from tests import test_evaluation as fixtures

inference.offline_environment()
from models.eomt import EoMT
from timm.models.vision_transformer import VisionTransformer


def tiny_model():
    """Real 12-block EoMT path, 32px images, randomly initialized 16-wide ViT."""
    encoder = torch.nn.Module()
    encoder.backbone = VisionTransformer(img_size=32, patch_size=16, embed_dim=16,
                                        depth=12, num_heads=2, reg_tokens=4, num_classes=0)
    encoder.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1))
    encoder.register_buffer("pixel_std", torch.ones(1, 3, 1, 1))
    model = torch.nn.Module()
    model.network = EoMT(encoder, num_classes=133, num_q=200, num_blocks=3,
                         masked_attn_enabled=False)
    model.mask_thresh = model.overlap_thresh = .8
    return model.eval()


class InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(11)
        self.model = tiny_model()
        self.image = torch.rand(1, 3, 32, 32)

    def forward_lengths(self, model):
        lengths = []
        hooks = [block.norm1.register_forward_pre_hook(
            lambda _, values: lengths.append(values[0].shape[1]))
            for block in model.network.encoder.backbone.blocks]
        try:
            with torch.inference_mode():
                outputs = model.network(self.image)
        finally:
            for hook in hooks:
                hook.remove()
        return lengths, outputs

    def test_disabled_baseline_exact_equivalence_and_no_weight_changes(self):
        original = copy.deepcopy(self.model)
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        method = self.model.network._disable_attn_mask.__func__
        record = apply_intervention(self.model, Intervention())
        first_lengths, first = self.forward_lengths(original)
        lengths, second = self.forward_lengths(self.model)
        self.assertEqual(lengths, first_lengths)
        self.assertEqual(record["query_block_indices_zero_based"], [9, 10, 11])
        self.assertIs(self.model.network._disable_attn_mask.__func__, method)
        self.assertFalse(self.model.network.masked_attn_enabled)
        for layers1, layers2 in zip(first, second):
            self.assertEqual(len(layers1), 1)
            torch.testing.assert_close(layers1[-1], layers2[-1], rtol=0, atol=0)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_query_removal_changes_tokens_before_forward_and_preserves_rows(self):
        before = self.model.network.q.weight.detach().clone()
        record = apply_intervention(self.model, Intervention(queries=100))
        self.assertEqual(self.model.network.num_q, 100)
        self.assertEqual(self.model.network.q.num_embeddings, 100)
        torch.testing.assert_close(self.model.network.q.weight, before[record["retained_query_ids"]])
        lengths, outputs = self.forward_lengths(self.model)
        self.assertEqual(lengths, [9] * 9 + [109] * 3)
        self.assertEqual(outputs[0][-1].shape[1], 100)
        self.assertEqual(outputs[1][-1].shape[1], 100)
        other = tiny_model()
        same = apply_intervention(other, Intervention(queries=100))
        self.assertEqual(record["retained_query_ids"], same["retained_query_ids"])
        different = apply_intervention(tiny_model(), Intervention(queries=100, query_seed=1))
        self.assertNotEqual(record["retained_query_ids"], different["retained_query_ids"])

    def test_later_insertion_preserves_all_encoder_blocks_and_buffer(self):
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        record = apply_intervention(self.model, Intervention(query_blocks=2))
        lengths, outputs = self.forward_lengths(self.model)
        self.assertEqual(lengths, [9] * 10 + [209] * 2)
        self.assertEqual(record["query_block_indices_zero_based"], [10, 11])
        self.assertEqual(self.model.network.attn_mask_probs.numel(), 3)
        self.assertIsNone(record["mask_probabilities"])
        self.assertEqual(outputs[0][-1].shape[1], 200)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_mask_endpoints_override_checkpoint_probabilities_and_only_query_patch(self):
        for mode, probability in (("zero", 0), ("full", 1)):
            model = tiny_model()
            model.network.attn_mask_probs.fill_(.37)
            record = apply_intervention(model, Intervention(attention=mode))
            self.assertEqual(record["mask_probabilities"], [probability] * 3)
            network = model.network
            logits = torch.full((1, 200, 8, 8), -1.)
            logits[:, :, :, 4:] = 1
            x = torch.zeros(1, 209, 16)
            # Even a forced rand==0 cannot leave zero-control restrictions.
            with patch("torch.rand", side_effect=AssertionError("no RNG at endpoints")):
                for block_index in (9, 10, 11):
                    mask = network._attn_mask(x, logits, block_index)
                    expected = torch.ones_like(mask)
                    if mode == "full":
                        expected[:, :200, 205:] = torch.tensor([False, True, False, True])
                    self.assertTrue(torch.equal(mask, expected))

    def test_zero_control_final_logits_match_baseline_and_full_forward_uses_masks(self):
        baseline = copy.deepcopy(self.model)
        apply_intervention(self.model, Intervention(attention="zero"))
        _, plain = self.forward_lengths(baseline)
        rng_before = torch.get_rng_state().clone()
        _, zero = self.forward_lengths(self.model)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertEqual(len(zero[0]), 4)
        for reference, actual in zip(plain, zero):
            torch.testing.assert_close(reference[-1], actual[-1], rtol=1e-5, atol=1e-6)
        full = tiny_model()
        apply_intervention(full, Intervention(attention="full"))
        seen = []
        original = full.network._attn

        def capture(module, x, mask, rope):
            seen.append(None if mask is None else mask.detach().clone())
            return original(module, x, mask, rope)

        with patch.object(full.network, "_attn", side_effect=capture):
            _, outputs = self.forward_lengths(full)
        self.assertEqual(sum(mask is not None for mask in seen), 3)
        self.assertEqual(len(outputs[0]), 4)
        # Shape/finite checks exercise the actual upstream attention with masks.
        self.assertTrue(torch.isfinite(outputs[0][-1]).all())

    def test_reject_combined_invalid_training_and_repeated_interventions(self):
        for setting in (Intervention(queries=50), Intervention(query_blocks=0),
                        Intervention(attention="random"), Intervention(query_seed=-1),
                        Intervention(queries=100, query_blocks=2)):
            with self.assertRaises(ValueError):
                apply_intervention(self.model, setting)
        self.model.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            apply_intervention(self.model, Intervention())
        self.model.eval()
        apply_intervention(self.model, Intervention())
        with self.assertRaisesRegex(ValueError, "fresh model"):
            apply_intervention(self.model, Intervention())


class WorkflowTests(unittest.TestCase):
    def test_500_image_selection_exact_reproducible_order_independent_and_seeded(self):
        ids = list(range(5000))
        expected = sorted(np.random.default_rng(0).choice(sorted(ids), 500, replace=False).tolist())
        self.assertEqual(select_ids(ids), expected)
        self.assertEqual(select_ids(list(reversed(ids))), expected)
        self.assertNotEqual(select_ids(ids, seed=1), expected)
        self.assertEqual(len(expected), len(set(expected)))
        for count in (0, 5001):
            with self.assertRaises(ValueError):
                select_ids(ids, count)

    def test_synthetic_evaluation_every_arm_provenance_and_isolated_outputs(self):
        with temporary_directory() as directory:
            root = Path(directory)
            ann, images, gt, data = fixtures.EvaluationWorkflowTests().make_dataset(root)
            checkpoint = root / "fixture.bin"
            checkpoint.write_bytes(b"synthetic fixture; never deserialized")
            before = checkpoint.read_bytes()
            output_root = inference.project_path("results/ablations") / root.name
            settings = [Intervention(), Intervention(queries=100), Intervention(query_blocks=2),
                        Intervention(attention="zero"), Intervention(attention="full")]
            try:
                for index, setting in enumerate(settings):
                    args = SimpleNamespace(images=str(images), panoptic=str(gt), annotations=str(ann),
                        checkpoint=str(checkpoint), output_dir=str(output_root / str(index)),
                        queries=setting.queries, query_blocks=setting.query_blocks,
                        attention=setting.attention, query_seed=0, sample_size=1,
                        sample_seed=0, threads=1, limit=1, full=False)
                    with patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
                        result = runner.evaluate(args, loader=lambda _: tiny_model(),
                            predictor=lambda model, image: np.zeros((2, 2, 2), dtype=np.int32), mapping={0: 1})
                    self.assertEqual(result["image_ids"], select_ids([2, 9], 1))
                    self.assertEqual(result["metrics"]["All"]["pq"], 1.)
                    self.assertEqual(result["checkpoint_sha256"], inference.file_sha256(checkpoint))
                    self.assertEqual(result["experiment"]["effective"]["queries"], setting.queries)
                    self.assertEqual(result["experiment"]["selection"]["seed"], 0)
                    self.assertIn("torch", result["runtime"])
                    self.assertIn("source_hashes", result["experiment"])
                    self.assertGreaterEqual(result["total_evaluation_seconds"], result["inference_seconds"])
                    manifest_path = Path(args.output_dir) / "experiment.json"
                    manifest = json.loads(manifest_path.read_text())
                    self.assertEqual(manifest["status"], "complete")
                    self.assertEqual(manifest["image_ids"], result["image_ids"])
                    self.assertTrue((manifest_path.parent / "predictions.json").is_file())
                    snapshot = manifest_path.read_bytes()
                    with self.assertRaisesRegex(ValueError, "already exists"):
                        runner.evaluate(args, loader=lambda _: self.fail("must not load on overwrite"))
                    self.assertEqual(manifest_path.read_bytes(), snapshot)
                self.assertEqual(checkpoint.read_bytes(), before)
                for forbidden in ("results/evaluation/coco-val5000", "results/ablations",
                                  "results/ablations/../outside"):
                    args.output_dir = forbidden
                    with self.assertRaisesRegex(ValueError, "Output must be under"):
                        runner.evaluate(args, loader=lambda _: self.fail("must not load outside root"))
            finally:
                if output_root.exists():
                    if output_root.resolve().parent != inference.project_path("results/ablations"):
                        raise RuntimeError("Refusing cleanup outside fixture root")
                    shutil.rmtree(output_root)

    def test_cli_rejects_combined_arms_before_loading(self):
        with patch.object(inference, "load_model", side_effect=AssertionError("must not load")):
            with patch("sys.stderr", new_callable=io.StringIO) as error:
                code = runner.main(["--output-dir", "results/ablations/never-created",
                                    "--queries", "100", "--attention", "full"])
        self.assertEqual(code, 2)
        self.assertIn("one factor", error.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
