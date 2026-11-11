"""Execute the five authorized 500-image arms sequentially, stopping on failure."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def execute(command, log):
    count = 0
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            if line.startswith("Evaluated "):
                count += 1
                if count % 25 == 0:
                    print(f"{log.stem}: {count}/500 images; {line.strip()}", flush=True)
            elif "Error" in line or "Traceback" in line or "Verified 500" in line:
                print(line.strip(), flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"Command failed with exit {code}; preserved log: {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/ablations/coco-val500", type=Path)
    parser.add_argument("--check-historical-integrity", action="store_true",
                        help="Require and hash the original baseline and smoke-test records")
    args = parser.parse_args()
    if __package__:
        from . import evaluate_panoptic as evaluation
        from .ablation_interventions import select_ids
    else:
        import evaluate_panoptic as evaluation
        from ablation_interventions import select_ids
    inference = evaluation.inference
    root = inference.project_path(args.output_dir)
    results = inference.project_path("results")
    allowed = results / "ablations"
    if root == allowed.resolve() or not root.is_relative_to(allowed.resolve()):
        raise ValueError("Suite output must be a new descendant of results/ablations")
    if root.exists():
        raise FileExistsError(f"Suite output already exists: {root}")
    checkpoint = inference.project_path(inference.DEFAULT_CHECKPOINT)
    inference.require_checkpoint(checkpoint)
    inference.offline_environment()
    if not inference.CONFIG.is_file():
        raise FileNotFoundError(f"Official configuration missing: {inference.CONFIG}")
    import importlib

    for module in ("torch", "torchvision", "timm", "lightning", "transformers", "yaml"):
        importlib.import_module(module)
    evaluation.index_dataset(
        inference.project_path("data/coco/annotations/panoptic_val2017.json"),
        inference.project_path("data/coco/val2017"),
        inference.project_path("data/coco/annotations/panoptic_val2017"),
        selector=lambda ids: select_ids(ids, 500, 0))
    protected = [checkpoint]
    if args.check_historical_integrity:
        protected.extend([results / "evaluation/coco-val5000/metrics.json",
                          results / "evaluation/coco-val5000/verification.json",
                          results / "ablations/smoke-test/summary.json"])
    original_hashes = {str(path): sha(path) for path in protected}
    if original_hashes[str(checkpoint)] != "7365b41a8a129afc74c12e5dd7af534146286d02822e4b89b7bbe8ea613ff74d":
        raise ValueError("Unexpected checkpoint hash")
    root.mkdir(parents=True, exist_ok=False)
    suite = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "historical_integrity_requested": args.check_historical_integrity,
             "protected_hashes_before": original_hashes, "free_disk_bytes_before": shutil.disk_usage(root).free,
             "arms": {}, "sample_size": 500, "sample_seed": 0, "threads": 4,
             "suite_source_sha256": sha(Path(__file__))}
    write(root / "suite.json", suite)
    started = time.perf_counter()
    scripts = Path(__file__).resolve().parent
    arms = [("baseline", []), ("queries100", ["--queries", "100", "--query-seed", "0"]),
            ("blocks2", ["--query-blocks", "2"]), ("mask-full", ["--attention", "full"]),
            ("mask-zero", ["--attention", "zero"])]
    try:
        for name, options in arms:
            command = [sys.executable, "-u", str(scripts / "evaluate_ablations.py"),
                       "--sample-size", "500", "--sample-seed", "0", "--threads", "4",
                       "--output-dir", str(root / name), *options]
            suite["arms"][name] = {"status": "running", "command": command,
                                    "started_at_utc": datetime.now(timezone.utc).isoformat()}
            write(root / "suite.json", suite)
            print(f"Starting {name}: exactly 500 images", flush=True)
            tick = time.perf_counter()
            execute(command, root / f"{name}.evaluation.log")
            suite["arms"][name]["process_wall_seconds"] = time.perf_counter() - tick
            execute([sys.executable, "-u", str(scripts / "verify_ablation.py"), str(root / name)],
                    root / f"{name}.verification.log")
            suite["arms"][name].update(status="verified", finished_at_utc=datetime.now(timezone.utc).isoformat())
            write(root / "suite.json", suite)
            print(f"{name}: evaluation and independent rescore PASSED", flush=True)
        records = []
        baseline = json.loads((root / "baseline/metrics.json").read_text())
        baseline_manifest = json.loads((root / "baseline/prediction_manifest.json").read_text())
        for name, _ in arms:
            metrics = json.loads((root / name / "metrics.json").read_text())
            verification = json.loads((root / name / "verification.json").read_text())
            if metrics["image_ids"] != baseline["image_ids"]:
                raise ValueError("Arms used different image sets")
            pq = 100 * metrics["metrics"]["All"]["pq"]
            difference = pq - 100 * baseline["metrics"]["All"]["pq"]
            records.append({"arm": name, "images": 500, "pq": pq,
                "sq": 100 * metrics["metrics"]["All"]["sq"], "rq": 100 * metrics["metrics"]["All"]["rq"],
                "things_pq": 100 * metrics["metrics"]["Things"]["pq"],
                "stuff_pq": 100 * metrics["metrics"]["Stuff"]["pq"],
                "signed_pq_difference": difference, "absolute_pq_difference": abs(difference),
                "evaluation_seconds": metrics["total_evaluation_seconds"], "inference_seconds": metrics["inference_seconds"],
                "mean_inference_seconds": verification["inference_mean_seconds"],
                "median_inference_seconds": verification["inference_median_seconds"],
                "p95_inference_seconds": verification["inference_p95_seconds"],
                "verification_seconds": verification["total_verification_seconds"], "independent_rescore": "passed"})
        zero_manifest = json.loads((root / "mask-zero/prediction_manifest.json").read_text())
        identical = sum(a["sha256"] == b["sha256"] for a, b in zip(baseline_manifest, zero_manifest))
        write(root / "comparison.json", {"metrics_scale": "0..100 metric points", "matched_image_ids": baseline["image_ids"],
              "rows": records, "zero_control_identical_pngs_to_baseline": identical,
              "limitation": "Fixed-checkpoint sensitivity study; not independently trained architectures"})
        with (root / "comparison.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        suite["protected_hashes_after"] = {str(path): sha(path) for path in protected}
        if suite["protected_hashes_after"] != original_hashes:
            raise ValueError("Protected artifact hash changed")
        suite.update(status="verified", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                     total_suite_seconds=time.perf_counter() - started, free_disk_bytes_after=shutil.disk_usage(root).free)
        write(root / "suite.json", suite)
        print(json.dumps(records, indent=2), flush=True)
    except Exception as error:
        suite.update(status="failed", error=repr(error), stopped_at_utc=datetime.now(timezone.utc).isoformat())
        write(root / "suite.json", suite)
        raise


if __name__ == "__main__":
    main()
