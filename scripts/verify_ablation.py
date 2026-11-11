"""Independent saved-mask verification/rescoring; no model or wrapper imports."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
from panopticapi.evaluation import pq_compute_single_core
from panopticapi.utils import rgb2id


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require(value, message):
    if not value:
        raise ValueError(message)


def close(a, b, tolerance=1e-12):
    require(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance, f"Mismatch: {a} vs {b}")


def verify(run):
    started = time.perf_counter()
    metrics = read(run / "metrics.json")
    manifest = read(run / "experiment.json")
    data_root = Path(os.environ.get("EOMT_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    gt_path = data_root / "coco/annotations/panoptic_val2017.json"
    gt_dir = data_root / "coco/annotations/panoptic_val2017"
    full = read(gt_path)
    images = {row["id"]: row for row in full["images"]}
    gt = {row["image_id"]: row for row in full["annotations"]}
    categories = {row["id"]: row for row in full["categories"]}
    require(len(images) == len(full["images"]) == len(gt) == len(full["annotations"]) == 5000, "Unexpected validation metadata")
    expected = sorted(int(i) for i in np.random.default_rng(0).choice(sorted(images), 500, replace=False))
    require(metrics["status"] == manifest["status"] == "complete", "Incomplete run")
    require(metrics["image_ids"] == manifest["image_ids"] == expected and metrics["evaluated_size"] == 500, "Incorrect sample")
    require(metrics["dataset_size"] == 5000, "Unexpected dataset size")
    require(metrics["annotations_sha256"] == sha(gt_path), "Annotation hash mismatch")
    require(sha(Path(metrics["checkpoint"])) == metrics["checkpoint_sha256"] == "7365b41a8a129afc74c12e5dd7af534146286d02822e4b89b7bbe8ea613ff74d", "Checkpoint mismatch")
    require(sha(Path(metrics["config"])) == metrics["config_sha256"], "Config mismatch")
    for source, digest in metrics["experiment"]["source_hashes"].items():
        require(sha(Path(__file__).resolve().parents[1] / source) == digest, "Source hash mismatch")
    subset = read(run / "ground_truth_subset.json")
    require(subset["images"] == [images[i] for i in expected], "Subset images mismatch")
    require(subset["annotations"] == [gt[i] for i in expected] and subset["categories"] == full["categories"], "Subset GT mismatch")
    prediction_json = read(run / "predictions.json")
    predictions = {row["image_id"]: row for row in prediction_json["annotations"]}
    require(len(predictions) == len(prediction_json["annotations"]) == 500 and sorted(predictions) == expected, "Prediction IDs mismatch")
    masks = run / "panoptic_predictions"
    require({p.name for p in masks.iterdir()} == {p["file_name"] for p in predictions.values()}, "Prediction files mismatch")
    times = list(csv.DictReader((run / "timings.csv").open(encoding="utf-8")))
    require(len(times) == len(metrics["per_image"]) == 500, "Timing coverage mismatch")
    png_manifest, seconds = [], []
    void_pixels = segment_count = 0
    for index, image_id in enumerate(expected):
        image, pred = images[image_id], predictions[image_id]
        path = (masks / pred["file_name"]).resolve()
        require(path.parent == masks.resolve(), "Mask path escape")
        with Image.open(path) as png:
            png.load()
            require(png.mode == "RGB" and png.size == (image["width"], image["height"]), "Invalid PNG dimensions/mode")
            ids = rgb2id(np.asarray(png))
        labels, counts = np.unique(ids, return_counts=True)
        areas = {int(k): int(v) for k, v in zip(labels, counts) if k != 0}
        segments = pred["segments_info"]
        require(len(segments) == len({s["id"] for s in segments}), "Duplicate segments")
        require(areas == {s["id"]: s["area"] for s in segments}, "Segment IDs/areas mismatch")
        require(all(0 < s["id"] < 256**3 and s["category_id"] in categories for s in segments), "Invalid segment metadata")
        void_pixels += int((ids == 0).sum())
        segment_count += len(segments)
        row, embedded = times[index], metrics["per_image"][index]
        require(int(row["image_id"]) == embedded["image_id"] == image_id, "Timing IDs mismatch")
        require(row["file_name"] == embedded["file_name"] == image["file_name"], "Timing filename mismatch")
        require(int(row["width"]) == embedded["width"] == image["width"] and int(row["height"]) == embedded["height"] == image["height"], "Timing dimensions mismatch")
        close(float(row["inference_seconds"]), embedded["inference_seconds"])
        require(float(row["inference_seconds"]) > 0, "Invalid inference duration")
        require(row["input_sha256"] == embedded["input_sha256"] == sha(data_root / "coco/val2017" / image["file_name"]), "Input hash mismatch")
        require(row["ground_truth_sha256"] == embedded["ground_truth_sha256"] == sha(gt_dir / gt[image_id]["file_name"]), "GT hash mismatch")
        seconds.append(float(row["inference_seconds"]))
        png_manifest.append({"image_id": image_id, "file_name": pred["file_name"], "sha256": sha(path), "segments": len(segments)})
    close(sum(seconds), metrics["inference_seconds"], 1e-8)
    require(metrics["total_evaluation_seconds"] > sum(seconds), "Invalid total duration")
    for row in csv.DictReader((run / "metrics.csv").open(encoding="utf-8")):
        for key in ("pq", "sq", "rq"):
            close(float(row[key]), metrics["metrics"][row["group"]][key])
    print("Verified 500 masks and provenance; independently rescoring saved predictions", flush=True)
    tick = time.perf_counter()
    stat = pq_compute_single_core(0, [(gt[i], copy.deepcopy(predictions[i])) for i in expected],
                                  str(gt_dir), str(masks), categories)
    rescored = {}
    for group, thing in (("All", None), ("Things", True), ("Stuff", False)):
        values, _ = stat.pq_average(categories, isthing=thing)
        rescored[group] = {key: float(values[key]) for key in ("pq", "sq", "rq")}
        rescored[group]["n"] = int(values["n"])
        require(values["n"] == metrics["metrics"][group]["n"], "Active category mismatch")
        for key in ("pq", "sq", "rq"):
            close(rescored[group][key], metrics["metrics"][group][key])
    class_rows = list(csv.DictReader((run / "per_class.csv").open(encoding="utf-8")))
    require(len(class_rows) == len(metrics["per_class"]) == len(categories) == 133, "Category coverage mismatch")
    for row in class_rows:
        category = int(row["category_id"])
        recorded = metrics["per_class"][str(category)]
        for key in ("tp", "fp", "fn"):
            require(getattr(stat[category], key) == recorded[key] == int(row[key]), "TP/FP/FN mismatch")
        close(stat[category].iou, recorded["iou_sum"], 1e-8)
        for key in ("pq", "sq", "rq", "iou_sum"):
            close(float(row[key]), recorded[key])
    report = {"status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "unique_images": 500, "prediction_pngs": len(png_manifest), "prediction_segments": segment_count,
              "void_pixels": void_pixels, "metrics": rescored, "independent_rescore_matches": True,
              "method": "separate process; direct unmodified COCO reference scorer; no model or evaluation-wrapper imports",
              "metric_absolute_tolerance": 1e-12, "rescore_seconds": time.perf_counter() - tick,
              "total_verification_seconds": time.perf_counter() - started,
              "inference_mean_seconds": float(np.mean(seconds)), "inference_median_seconds": float(np.median(seconds)),
              "inference_p95_seconds": float(np.percentile(seconds, 95)),
              "tp_total": sum(stat[k].tp for k in categories), "fp_total": sum(stat[k].fp for k in categories),
              "fn_total": sum(stat[k].fn for k in categories), "verifier_sha256": sha(Path(__file__)),
              "scorer_sha256": sha(Path(__file__).parent / "vendor/panopticapi/evaluation.py")}
    write_new(run / "rescored_metrics.json", {"method": report["method"], "metrics": rescored, "rescore_seconds": report["rescore_seconds"]})
    write_new(run / "prediction_manifest.json", png_manifest)
    write_new(run / "verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    verify(parser.parse_args().run)
