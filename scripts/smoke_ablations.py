"""Real-checkpoint, single-image smoke verification; never loops over a dataset."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import sys
import time
from types import MethodType

import numpy as np
from PIL import Image

if __package__:
    from . import evaluate_panoptic as evaluation
    from .ablation_interventions import Intervention, apply_intervention, select_ids
else:
    import evaluate_panoptic as evaluation
    from ablation_interventions import Intervention, apply_intervention, select_ids

inference = evaluation.inference
EXPECTED_CHECKPOINT = "7365b41a8a129afc74c12e5dd7af534146286d02822e4b89b7bbe8ea613ff74d"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def observe_prediction(model, image):
    """Observe the existing predictor without replacing preprocessing/forward/output."""
    network = model.network
    observations = {"block_input_tokens": [], "attention": []}
    final = {}
    hooks = [block.norm1.register_forward_pre_hook(
        lambda _, values: observations["block_input_tokens"].append(int(values[0].shape[1])))
        for block in network.encoder.backbone.blocks]

    def capture_outputs(_, inputs, outputs):
        observations["prediction_stages"] = len(outputs[0])
        final["mask_logits"] = outputs[0][-1].detach().clone()
        final["class_logits"] = outputs[1][-1].detach().clone()

    hooks.append(network.register_forward_hook(capture_outputs))
    original_attention = network._attn

    def attention(self, module, x, mask, rope):
        row = {"block_index": len(observations["attention"]), "mask_present": mask is not None}
        if mask is not None:
            q = self.num_q
            patch_start = q + self.encoder.backbone.num_prefix_tokens
            blocked = int((~mask[:, :q, patch_start:]).sum().item())
            row.update(blocked_query_patch_pairs=blocked,
                       query_patch_pairs=int(mask[:, :q, patch_start:].numel()),
                       blocked_other_pairs=int((~mask).sum().item()) - blocked)
        observations["attention"].append(row)
        return original_attention(module, x, mask, rope)

    network._attn = MethodType(attention, network)
    tick = time.perf_counter()
    try:
        prediction = inference.predict_image(model, image)
        seconds = time.perf_counter() - tick
    finally:
        del network._attn  # Restore class method resolution, not an instance override.
        for hook in hooks:
            hook.remove()
    return prediction, final, observations, seconds


def validate_observations(model, observations, setting):
    n = model.network
    patches = int(np.prod(n.encoder.backbone.patch_embed.grid_size))
    prefix = n.encoder.backbone.num_prefix_tokens
    depth = len(n.encoder.backbone.blocks)
    expected = [patches + prefix] * (depth - setting.query_blocks)
    expected += [patches + prefix + setting.queries] * setting.query_blocks
    require(observations["block_input_tokens"] == expected, "Unexpected query insertion/count")
    active = [row for row in observations["attention"] if row["mask_present"]]
    require(len(observations["attention"]) == depth, "Not all encoder blocks ran")
    require(len(active) == (3 if setting.attention != "off" else 0), "Wrong masking block count")
    require(observations["prediction_stages"] == (4 if active else 1), "Wrong prediction stage count")
    if setting.attention == "zero":
        require(all(row["blocked_query_patch_pairs"] == 0 for row in active), "M0 is not fully open")
    if setting.attention == "full":
        require(all(row["blocked_query_patch_pairs"] > 0 for row in active), "M1 restrictions ineffective")
    require(all(row["blocked_other_pairs"] == 0 for row in active), "Mask altered non-query-patch attention")


def save_and_validate(image, prediction, destination, metadata, mapping, categories):
    require(prediction.shape == (image.height, image.width, 2), "Invalid prediction dimensions")
    inference.save_prediction(image, prediction, destination, {**metadata, "category_mapping": mapping})
    segments = evaluation.export_prediction(prediction, mapping, categories, destination / "panoptic.png")
    evaluation.write_json(destination / "panoptic.json", {
        "image_id": metadata["image_id"], "file_name": "panoptic.png", "segments_info": segments})
    with Image.open(destination / "panoptic.png") as mask:
        require(mask.mode == "RGB" and mask.size == image.size, "Invalid exported PNG")
        ids = evaluation.rgb2id(np.asarray(mask))
    found_ids, counts = np.unique(ids, return_counts=True)
    found = {int(k): int(v) for k, v in zip(found_ids, counts) if k != 0}
    require(found == {s["id"]: s["area"] for s in segments}, "Segment IDs/areas disagree")
    require(all(s["category_id"] in categories for s in segments), "Unknown category")
    require(int((ids == 0).sum()) == int((prediction[..., 0] == 133).sum()), "Void area mismatch")
    record = json.loads((destination / "prediction.json").read_text())
    require(len(record["segments_info"]) == len(segments), "Raw/export segment count mismatch")
    for segment in record["segments_info"]:
        pixels = prediction[..., 1] == segment["id"]
        classes = np.unique(prediction[..., 0][pixels])
        require(len(classes) == 1 and int(classes[0]) == segment["class_id"], "Mixed segment classes")
        require(int(pixels.sum()) == segment["area"], "Raw segment area mismatch")
        require(mapping[segment["class_id"]] == segment["coco_category_id"], "Category mapping mismatch")
        require(bool(categories[segment["coco_category_id"]]["isthing"]) == segment["isthing"], "Thing flag mismatch")
    for filename in ("overlay.png", "segmentation.png"):
        with Image.open(destination / filename) as visual:
            visual.load()
            require(visual.size == image.size, "Invalid visualization dimensions")
    with np.load(destination / "prediction.npz", allow_pickle=False) as saved:
        require(np.array_equal(saved["semantic"], prediction[..., 0]), "Saved class map differs")
        require(np.array_equal(saved["segment_ids"], prediction[..., 1]), "Saved segment map differs")
    return {"status": "passed", "segments": len(segments),
            "category_ids": sorted({s["category_id"] for s in segments}),
            "void_pixels": int((ids == 0).sum()), "height": image.height, "width": image.width,
            "panoptic_png_sha256": inference.file_sha256(destination / "panoptic.png")}


def compare(reference_prediction, reference_logits, prediction, logits, exact=False):
    import torch

    result = {"semantic_pixels_equal": bool(np.array_equal(reference_prediction[..., 0], prediction[..., 0])),
              "segment_pixels_equal": bool(np.array_equal(reference_prediction[..., 1], prediction[..., 1])),
              "rtol": 0.0 if exact else 1e-5, "atol": 0.0 if exact else 1e-5}
    for key in ("mask_logits", "class_logits"):
        result[key + "_maximum_absolute_difference"] = float((reference_logits[key] - logits[key]).abs().max())
        result[key + "_close"] = bool(torch.allclose(
            reference_logits[key], logits[key], rtol=result["rtol"], atol=result["atol"]))
    result["passed"] = all(result[key] for key in (
        "semantic_pixels_equal", "segment_pixels_equal", "mask_logits_close", "class_logits_close"))
    return result


def run(output):
    import torch

    start = time.perf_counter()
    inference.offline_environment()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    checkpoint = inference.require_checkpoint(inference.project_path(inference.DEFAULT_CHECKPOINT))
    require(inference.file_sha256(checkpoint) == EXPECTED_CHECKPOINT, "Checkpoint hash differs")
    annotation_path = inference.project_path("data/coco/annotations/panoptic_val2017.json")
    data = json.loads(annotation_path.read_text())
    subset = select_ids([row["id"] for row in data["images"]], 500, 0)
    first = subset[0]  # Not a new independent sample of size 1.
    image_record = next(row for row in data["images"] if row["id"] == first)
    source_path = inference.project_path("data/coco/val2017") / image_record["file_name"]
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    require(image.size == (image_record["width"], image_record["height"]), "Input dimensions disagree")
    from datasets.coco_panoptic import CLASS_MAPPING
    mapping = {v: k for k, v in CLASS_MAPPING.items()}
    categories = {row["id"]: row for row in data["categories"]}
    require(set(categories) == set(CLASS_MAPPING), "Unexpected COCO category set")
    summary = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_id": first, "image_file": image_record["file_name"], "selected_images_executed": 1,
        "parent_subset_size": 500, "parent_subset_seed": 0, "parent_subset_ids": subset,
        "checkpoint_sha256": EXPECTED_CHECKPOINT, "input_sha256": inference.file_sha256(source_path),
        "annotations_sha256": inference.file_sha256(annotation_path),
        "config_sha256": inference.file_sha256(inference.CONFIG), "source_commit_expected": inference.COMMIT,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "precision": "float32",
                    "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()},
        "source_hashes": {p.name: inference.file_sha256(p) for p in (
            Path(__file__), Path(__file__).with_name("ablation_interventions.py"),
            Path(inference.__file__), Path(evaluation.__file__))},
        "command": list(sys.argv), "arms": {},
        "timing_scope": "single cold prediction call including read-only observation hooks and final-logit copies; excludes load/export",
    }
    evaluation.write_json(output / "summary.json", summary)
    reference_prediction = reference_logits = baseline_prediction = baseline_logits = None
    configurations = [("unmodified-reference", None), ("baseline", Intervention()),
                      ("queries100", Intervention(queries=100)),
                      ("blocks2", Intervention(query_blocks=2)),
                      ("mask-full", Intervention(attention="full")),
                      ("mask-zero", Intervention(attention="zero"))]
    for name, settings in configurations:
        tick = time.perf_counter()
        model = inference.load_model(checkpoint)
        load_seconds = time.perf_counter() - tick
        effective = apply_intervention(model, settings) if settings is not None else {"path": "unmodified inference.load_model/predict_image"}
        prediction, logits, observations, seconds = observe_prediction(model, image)
        validate_observations(model, observations, settings or Intervention())
        require(all(bool(torch.isfinite(value).all()) for value in logits.values()), "Non-finite final logits")
        validation = save_and_validate(image, prediction, output / name,
            {"image_id": first, "checkpoint_sha256": EXPECTED_CHECKPOINT, "inference_seconds": seconds,
             "model_load_seconds": load_seconds, "intervention": effective}, mapping, categories)
        row = {"status": "passed", "model_load_seconds": load_seconds, "inference_seconds": seconds,
               "effective": effective, "observed": observations, "validation": validation,
               "final_mask_logits_shape": list(logits["mask_logits"].shape),
               "final_class_logits_shape": list(logits["class_logits"].shape)}
        if name == "unmodified-reference":
            reference_prediction, reference_logits = prediction, logits
        elif name == "baseline":
            row["comparison_to_unmodified"] = compare(reference_prediction, reference_logits, prediction, logits, exact=True)
            require(row["comparison_to_unmodified"]["passed"], "Baseline differs from unmodified path")
            baseline_prediction, baseline_logits = prediction, logits
        else:
            row["semantic_pixels_different_from_baseline"] = int(np.count_nonzero(prediction[..., 0] != baseline_prediction[..., 0]))
            if name == "mask-zero":
                row["comparison_to_baseline"] = compare(baseline_prediction, baseline_logits, prediction, logits)
                require(row["comparison_to_baseline"]["passed"], "Zero control differs beyond tolerance")
        summary["arms"][name] = row
        evaluation.write_json(output / name / "verification.json", row)
        evaluation.write_json(output / "summary.json", summary)
        print(f"{name}: PASS; {seconds:.6f}s; {validation['segments']} segments", flush=True)
        del model, prediction, logits
        gc.collect()
    require(inference.file_sha256(checkpoint) == EXPECTED_CHECKPOINT, "Checkpoint changed")
    summary.update(status="passed", checkpoint_hash_after=EXPECTED_CHECKPOINT,
                   finished_at_utc=datetime.now(timezone.utc).isoformat(),
                   total_seconds=time.perf_counter() - start)
    evaluation.write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/ablations/smoke-test")
    args = parser.parse_args()
    output = inference.project_path(args.output_dir)
    root = inference.project_path("results/ablations")
    if output == root or not output.is_relative_to(root):
        parser.error("Output must be a new descendant of results/ablations")
    # Fail before reading a model or executing inference if the path exists.
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(output)
    except Exception as error:
        evaluation.write_json(output / "failure.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
