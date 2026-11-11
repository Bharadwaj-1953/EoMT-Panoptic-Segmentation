"""Offline COCO 2017 PQ evaluation with the official COCO reference scorer."""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image

if __package__:
    from . import infer_panoptic as inference
else:
    import infer_panoptic as inference

VENDOR = Path(__file__).resolve().parent / "vendor"
sys.path.insert(0, str(VENDOR))
from panopticapi.evaluation import PQStat, pq_compute_single_core
from panopticapi.utils import id2rgb, rgb2id

SCORER_REVISION = "7bb4655548f98f3fedc07bf37e9040a992b054b0"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def child_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset filename escapes its directory: {name}")
    return path


def index_dataset(annotation_path, image_dir, panoptic_dir, limit=10, full=False, selector=None):
    """Validate metadata and selected file presence before allocating a model."""
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {x["id"]: x for x in data["images"]}
    annotations = {x["image_id"]: x for x in data["annotations"]}
    categories = {x["id"]: x for x in data["categories"]}
    if len(images) != len(data["images"]) or len(annotations) != len(data["annotations"]):
        raise ValueError("Duplicate image or annotation IDs")
    if not images or images.keys() != annotations.keys():
        raise ValueError("Images and annotations must have identical nonempty ID sets")
    if len(categories) != len(data["categories"]) or not categories:
        raise ValueError("Empty or duplicate categories")
    if any(x.get("isthing") not in (0, 1) for x in categories.values()):
        raise ValueError("Each category needs isthing=0 or 1")
    if full and len(images) != 5000:
        raise ValueError("--full requires exactly 5,000 COCO val2017 images")
    if limit <= 0:
        raise ValueError("--limit must be positive")
    selected = sorted(images) if full else sorted(images)[:limit]
    if selector is not None:
        selected = selector(sorted(images))
        if (not selected or selected != sorted(set(selected))
                or any(i not in images for i in selected)):
            raise ValueError("Selected IDs must be nonempty, unique, sorted and present in the dataset")
    for image_id in selected:
        im, ann = images[image_id], annotations[image_id]
        for path in (child_path(image_dir, im["file_name"]), child_path(panoptic_dir, ann["file_name"])):
            if not path.is_file():
                raise FileNotFoundError(f"Required local COCO file missing: {path}; no download attempted")
        segments = ann["segments_info"]
        if len({s["id"] for s in segments}) != len(segments):
            raise ValueError("Duplicate ground-truth segment IDs")
        for segment in segments:
            if not 0 < segment["id"] < 256**3 or segment["category_id"] not in categories:
                raise ValueError("Invalid ground-truth segment or category ID")
            if segment.get("iscrowd") not in (0, 1) or segment.get("area", 0) <= 0:
                raise ValueError("Ground-truth segments need iscrowd and positive area")
    return data, images, annotations, categories, selected


def validate_ground_truth(path, annotation, size):
    with Image.open(path) as source:
        if source.mode != "RGB" or source.size != size:
            raise ValueError("Ground-truth PNG must be RGB and match image dimensions")
        ids = rgb2id(np.asarray(source))
    labels, counts = np.unique(ids, return_counts=True)
    areas = {int(k): int(v) for k, v in zip(labels, counts) if k != 0}
    expected = {s["id"]: s["area"] for s in annotation["segments_info"]}
    if areas != expected:
        raise ValueError("Ground-truth PNG IDs/areas disagree with segments_info")


def export_prediction(prediction, mapping, categories, path):
    """Convert contiguous EoMT labels to COCO category IDs and RGB segment IDs."""
    if prediction.ndim != 3 or prediction.shape[-1] != 2 or not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError("Expected integer H x W x 2 predictions")
    semantic, instances = prediction[..., 0], prediction[..., 1]
    void = semantic == 133
    if np.any((semantic < 0) | (semantic > 133)) or not np.array_equal(void, instances == -1):
        raise ValueError("Invalid prediction class/void encoding")
    if np.any(instances < -1):
        raise ValueError("Invalid negative segment ID")
    ids = np.zeros(semantic.shape, dtype=np.int32)
    segments = []
    for new_id, old_id in enumerate(np.unique(instances[~void]), 1):
        mask = instances == old_id
        labels = np.unique(semantic[mask])
        if len(labels) != 1 or int(labels[0]) not in mapping:
            raise ValueError("Segment has mixed or unmapped classes")
        category = mapping[int(labels[0])]
        if category not in categories or new_id >= 256**3:
            raise ValueError("Unknown category or excessive segment count")
        ids[mask] = new_id
        segments.append({"id": new_id, "category_id": category, "area": int(mask.sum())})
    Image.fromarray(id2rgb(ids)).save(path)
    return segments


def summarize(stat, categories):
    """Use reference category averaging; undefined empty groups become null."""
    per_class = {}
    for category, info in categories.items():
        s = stat[category]
        denominator = s.tp + 0.5 * s.fp + 0.5 * s.fn
        per_class[category] = {"name": info.get("name", str(category)), "isthing": info["isthing"],
                               "pq": s.iou / denominator if denominator else 0.0,
                               "sq": s.iou / s.tp if s.tp else 0.0,
                               "rq": s.tp / denominator if denominator else 0.0,
                               "tp": int(s.tp), "fp": int(s.fp), "fn": int(s.fn),
                               "iou_sum": float(s.iou), "included": bool(denominator)}
    groups = {}
    for name, thing in (("All", None), ("Things", 1), ("Stuff", 0)):
        active = [x for x in per_class.values() if x["included"] and (thing is None or x["isthing"] == thing)]
        if active:
            reference, _ = stat.pq_average(categories, isthing=None if thing is None else bool(thing))
            groups[name] = {k: float(reference[k]) for k in ("pq", "sq", "rq")}
            groups[name]["n"] = reference["n"]
        else:
            groups[name] = {"pq": None, "sq": None, "rq": None, "n": 0}
    return groups, per_class


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args, *, loader=None, predictor=None, mapping=None, selector=None,
             experiment=None, output_root="results/evaluation"):
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    image_dir = inference.project_path(args.images)
    gt_dir = inference.project_path(args.panoptic)
    annotation_path = inference.project_path(args.annotations)
    output = inference.project_path(args.output_dir)
    allowed_root = inference.project_path(output_root)
    if output == allowed_root or not output.is_relative_to(allowed_root):
        raise ValueError(f"Output must be under {output_root}/")
    if output.exists():
        raise ValueError("Output already exists; choose a new run directory")
    checkpoint = inference.require_checkpoint(inference.project_path(args.checkpoint))
    data, images, annotations, categories, selected = index_dataset(
        annotation_path, image_dir, gt_dir, args.limit, args.full, selector=selector)
    inference.offline_environment()
    if mapping is None:
        from datasets.coco_panoptic import CLASS_MAPPING
        mapping = {v: k for k, v in CLASS_MAPPING.items()}
        if set(categories) != set(CLASS_MAPPING) or any(
            bool(categories[cat]["isthing"]) != (cls < 80) for cat, cls in CLASS_MAPPING.items()
        ):
            raise ValueError("Expected official 133-category COCO panoptic metadata")
    load_started = time.perf_counter()
    model = (loader or inference.load_model)(checkpoint)
    load_seconds = time.perf_counter() - load_started
    output.mkdir(parents=True)
    if experiment is not None:
        write_json(output / "experiment.json", {
            "status": "running", "started_at_utc": started_at,
            "experiment": experiment, "image_ids": selected,
            "checkpoint_sha256": inference.file_sha256(checkpoint),
            "annotations_sha256": inference.file_sha256(annotation_path)})
    pred_dir = output / "panoptic_predictions"
    pred_dir.mkdir()
    stat = PQStat()
    timings, pred_annotations = [], []
    for image_id in selected:
        image_record, annotation = images[image_id], annotations[image_id]
        image_path = child_path(image_dir, image_record["file_name"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if image.size != (image_record["width"], image_record["height"]):
            raise ValueError(f"Image dimensions disagree with metadata: {image_id}")
        validate_ground_truth(child_path(gt_dir, annotation["file_name"]), annotation, image.size)
        tick = time.perf_counter()
        prediction = (predictor or inference.predict_image)(model, image)
        seconds = time.perf_counter() - tick
        if prediction.shape[:2] != (image.height, image.width):
            raise ValueError("Prediction dimensions disagree with input")
        name = f"{image_id:012d}.png"
        segments = export_prediction(prediction, mapping, categories, pred_dir / name)
        pred_ann = {"image_id": image_id, "file_name": name, "segments_info": segments}
        # The reference scorer replaces prediction areas with NumPy scalars.
        # Keep export metadata independent of those in-place mutations.
        stat += pq_compute_single_core(0, [(annotation, copy.deepcopy(pred_ann))], str(gt_dir), str(pred_dir), categories)
        pred_annotations.append(pred_ann)
        timings.append({"image_id": image_id, "file_name": image_record["file_name"],
                        "width": image.width, "height": image.height, "inference_seconds": seconds,
                        "input_sha256": inference.file_sha256(image_path),
                        "ground_truth_sha256": inference.file_sha256(child_path(gt_dir, annotation["file_name"]))})
        print(f"Evaluated {image_id}: {seconds:.3f}s", flush=True)
    groups, per_class = summarize(stat, categories)
    write_json(output / "predictions.json", {"annotations": pred_annotations})
    write_json(output / "ground_truth_subset.json", {**data, "images": [images[i] for i in selected],
                                                    "annotations": [annotations[i] for i in selected]})
    write_csv(output / "timings.csv", timings, list(timings[0]))
    write_csv(output / "metrics.csv", [{"group": k, **v} for k, v in groups.items()], ["group", "pq", "sq", "rq", "n"])
    write_csv(output / "per_class.csv", [{"category_id": k, **v} for k, v in per_class.items()],
              ["category_id", *next(iter(per_class.values()))])
    result = {"status": "complete", "model_id": inference.MODEL_ID, "device": "cpu", "precision": "float32",
              "checkpoint": str(checkpoint), "checkpoint_sha256": inference.file_sha256(checkpoint),
              "config": str(inference.CONFIG), "config_sha256": inference.file_sha256(inference.CONFIG),
              "source_commit_expected": inference.COMMIT, "scorer_revision": SCORER_REVISION,
              "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                          "torch": importlib.metadata.version("torch"),
                          "numpy": np.__version__},
              "scorer": "official cocodataset/panopticapi pq_compute_single_core; serial accumulation",
              "annotations": str(annotation_path), "annotations_sha256": inference.file_sha256(annotation_path),
              "images_dir": str(image_dir), "panoptic_dir": str(gt_dir),
              "dataset_size": len(images), "evaluated_size": len(selected), "image_ids": selected,
              "selection": "all" if args.full else "ascending image ID, first N",
              "requested_limit": None if args.full else args.limit,
              "mask_threshold": model.mask_thresh, "overlap_threshold": model.overlap_thresh,
              "model_load_seconds": load_seconds, "inference_seconds": sum(x["inference_seconds"] for x in timings),
              "metrics_scale": "0..1 (multiply by 100 for reported PQ points)",
              "metrics": groups, "per_class": per_class, "per_image": timings}
    result["started_at_utc"] = started_at
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    if experiment is not None:
        result["experiment"] = experiment
        result["selection"] = experiment["selection"]
        result["requested_limit"] = len(selected)
    result["total_evaluation_seconds"] = time.perf_counter() - started
    write_json(output / "metrics.json", result)
    if experiment is not None:
        write_json(output / "experiment.json", {
            "status": "complete", "experiment": experiment,
            **{key: result[key] for key in (
                "image_ids", "checkpoint_sha256", "annotations_sha256", "config_sha256",
                "source_commit_expected", "scorer_revision", "runtime", "started_at_utc",
                "finished_at_utc", "model_load_seconds", "inference_seconds",
                "total_evaluation_seconds")}})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="data/coco/val2017")
    parser.add_argument("--panoptic", default="data/coco/annotations/panoptic_val2017")
    parser.add_argument("--annotations", default="data/coco/annotations/panoptic_val2017.json")
    parser.add_argument("--checkpoint", default=inference.DEFAULT_CHECKPOINT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int, default=10)
    group.add_argument("--full", action="store_true")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--output-dir", default="results/evaluation/coco-val2017-first10")
    args = parser.parse_args(argv)
    try:
        result = evaluate(args)
        print(json.dumps(result["metrics"], indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        print(f"Evaluation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
