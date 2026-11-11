"""Verify all full-run artifacts and rescore masks using COCO's reference API.

Run from the project root: python -m scripts.verification.verify_and_rescore
This does not import the model or the project's evaluation wrapper.
"""
from pathlib import Path
import copy
import os
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(os.environ["EOMT_VERIFY_RUN"]).resolve()
OUT = Path(os.environ["EOMT_VERIFY_OUTPUT"]).resolve()
if OUT == RUN or OUT.is_relative_to(RUN):
    raise ValueError("Verification output must be separate from the saved run")
OUT.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(ROOT / "scripts/vendor"))
from panopticapi.evaluation import pq_compute_single_core
from panopticapi.utils import rgb2id


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def close(a, b, tolerance=1e-12):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=0, abs_tol=tolerance), (a, b)


started = time.perf_counter()
metrics = read_json(RUN / "metrics.json")
gt_path = ROOT / "data/coco/annotations/panoptic_val2017.json"
gt = read_json(gt_path)
subset = read_json(RUN / "ground_truth_subset.json")
pred = read_json(RUN / "predictions.json")
images = {x["id"]: x for x in gt["images"]}
annotations = {x["image_id"]: x for x in gt["annotations"]}
predictions = {x["image_id"]: x for x in pred["annotations"]}
categories = {x["id"]: x for x in gt["categories"]}
expected = sorted(images)
assert len(images) == len(gt["images"]) == len(annotations) == len(gt["annotations"]) == 5000
assert len(predictions) == len(pred["annotations"]) == 5000
assert sorted(predictions) == sorted(annotations) == expected
assert metrics["status"] == "complete" and metrics["selection"] == "all"
assert metrics["dataset_size"] == metrics["evaluated_size"] == 5000
assert metrics["image_ids"] == expected and len(set(metrics["image_ids"])) == 5000
assert metrics["device"] == "cpu" and metrics["precision"] == "float32"
assert metrics["model_id"] == "tue-mps/coco_panoptic_eomt_small_640_2x"
assert {x["id"]: x for x in subset["images"]} == images
assert {x["image_id"]: x for x in subset["annotations"]} == annotations
assert subset["categories"] == gt["categories"]
assert metrics["annotations_sha256"] == sha(gt_path)
assert metrics["checkpoint_sha256"] == sha(ROOT / "checkpoints/coco_panoptic_eomt_small_640_2x/pytorch_model.bin") == "7365b41a8a129afc74c12e5dd7af534146286d02822e4b89b7bbe8ea613ff74d"
assert metrics["config_sha256"] == sha(ROOT / "upstream/eomt/configs/dinov2/coco/panoptic/eomt_small_640_2x.yaml")

timings = list(csv.DictReader((RUN / "timings.csv").open(encoding="utf-8")))
assert len(timings) == len(metrics["per_image"]) == 5000
assert [int(x["image_id"]) for x in timings] == expected
assert [x["image_id"] for x in metrics["per_image"]] == expected
mask_dir = RUN / "panoptic_predictions"
assert len(list(mask_dir.glob("*.png"))) == 5000
assert {p.name for p in mask_dir.iterdir()} == {x["file_name"] for x in predictions.values()}
mask_manifest = []
times = []
void_pixels = 0
segment_count = 0
for index, image_id in enumerate(expected):
    image, annotation, prediction = images[image_id], annotations[image_id], predictions[image_id]
    mask_path = mask_dir / prediction["file_name"]
    with Image.open(mask_path) as mask:
        mask.load()
        assert mask.mode == "RGB" and mask.size == (image["width"], image["height"])
        ids = rgb2id(np.asarray(mask))
        labels, counts = np.unique(ids, return_counts=True)
        areas = {int(k): int(v) for k, v in zip(labels, counts) if k != 0}
        void_pixels += int((ids == 0).sum())
    segments = prediction["segments_info"]
    assert len({s["id"] for s in segments}) == len(segments)
    assert areas == {s["id"]: s["area"] for s in segments}
    assert all(0 < s["id"] < 256**3 and s["category_id"] in categories for s in segments)
    assert sum(areas.values()) <= image["width"] * image["height"]
    segment_count += len(segments)
    row, embedded = timings[index], metrics["per_image"][index]
    assert row["file_name"] == embedded["file_name"] == image["file_name"]
    assert int(row["width"]) == embedded["width"] == image["width"]
    assert int(row["height"]) == embedded["height"] == image["height"]
    seconds = float(row["inference_seconds"])
    assert math.isfinite(seconds) and seconds > 0
    assert seconds == embedded["inference_seconds"]
    assert row["input_sha256"] == embedded["input_sha256"] == sha(ROOT / "data/coco/val2017" / image["file_name"])
    assert row["ground_truth_sha256"] == embedded["ground_truth_sha256"] == sha(ROOT / "data/coco/annotations/panoptic_val2017" / annotation["file_name"])
    times.append(seconds)
    mask_manifest.append({"image_id": image_id, "file_name": prediction["file_name"],
                          "sha256": sha(mask_path), "segments": len(segments)})
    if (index + 1) % 500 == 0:
        print(f"Verified {index + 1}/5000 output masks and timing/provenance records", flush=True)

close(sum(times), metrics["inference_seconds"], 1e-8)
assert metrics["total_evaluation_seconds"] > metrics["inference_seconds"] + metrics["model_load_seconds"]
for row in csv.DictReader((RUN / "metrics.csv").open(encoding="utf-8")):
    group = metrics["metrics"][row["group"]]
    assert int(row["n"]) == group["n"]
    for key in ("pq", "sq", "rq"):
        close(float(row[key]), group[key])
class_rows = list(csv.DictReader((RUN / "per_class.csv").open(encoding="utf-8")))
assert len(class_rows) == len(metrics["per_class"]) == len(categories) == 133
for row in class_rows:
    recorded = metrics["per_class"][row["category_id"]]
    for key in ("pq", "sq", "rq", "iou_sum"):
        close(float(row[key]), recorded[key])
    for key in ("tp", "fp", "fn"):
        assert int(row[key]) == recorded[key]

print("Rescoring all 5,000 exported masks with the unmodified COCO reference worker", flush=True)
rescore_started = time.perf_counter()
stat = pq_compute_single_core(0, [(annotations[i], copy.deepcopy(predictions[i])) for i in expected],
                              str(ROOT / "data/coco/annotations/panoptic_val2017"), str(mask_dir), categories)
rescored = {}
maximum_class_difference = 0.0
for group_name, thing in (("All", None), ("Things", True), ("Stuff", False)):
    group, per_class = stat.pq_average(categories, isthing=thing)
    native = {k: float(group[k]) for k in ("pq", "sq", "rq")}
    native["n"] = int(group["n"])
    rescored[group_name] = native
    assert native["n"] == metrics["metrics"][group_name]["n"]
    for key in ("pq", "sq", "rq"):
        close(native[key], metrics["metrics"][group_name][key])
        assert 0 <= native[key] <= 1
    if group_name == "All":
        for category, values in per_class.items():
            recorded = metrics["per_class"][str(category)]
            for key in ("pq", "sq", "rq"):
                close(float(values[key]), recorded[key])
            maximum_class_difference = max(maximum_class_difference, abs(stat[category].iou - recorded["iou_sum"]))
            close(stat[category].iou, recorded["iou_sum"], 1e-8)
            for key in ("tp", "fp", "fn"):
                assert getattr(stat[category], key) == recorded[key]

rescore_seconds = time.perf_counter() - rescore_started
write_json(OUT / "rescored_metrics.json", {"method": "separate process; direct unmodified COCO reference worker and PQStat.pq_average; no model/wrapper imports",
                                           "metrics": rescored, "rescore_seconds": rescore_seconds})
write_json(OUT / "prediction_manifest.json", mask_manifest)
report = {"status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
          "unique_images": 5000, "prediction_pngs": len(mask_manifest), "prediction_segments": segment_count,
          "void_pixels": void_pixels, "image_annotation_prediction_id_sets_equal": True,
          "all_masks_decode_and_match_dimensions": True, "all_segment_ids_areas_categories_valid": True,
          "all_input_ground_truth_checkpoint_config_hashes_match": True, "csv_json_metrics_timings_agree": True,
          "independent_rescore_matches": True, "metric_absolute_tolerance": 1e-12,
          "class_iou_sum_absolute_tolerance": 1e-8, "maximum_class_iou_sum_difference": float(maximum_class_difference),
          "tp_total": sum(stat[k].tp for k in categories), "fp_total": sum(stat[k].fp for k in categories),
          "fn_total": sum(stat[k].fn for k in categories), "metrics": rescored,
          "inference_mean_seconds": float(np.mean(times)), "inference_median_seconds": float(np.median(times)),
          "inference_p95_seconds": float(np.percentile(times, 95)), "inference_min_seconds": min(times),
          "inference_max_seconds": max(times), "total_verification_seconds": time.perf_counter() - started,
          "rescore_seconds": rescore_seconds}
write_json(OUT / "verification.json", report)
print(json.dumps(report, indent=2), flush=True)
