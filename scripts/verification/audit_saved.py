"""Local saved-artifact audit. Never imports a model or evaluates predictions anew."""
import argparse
import ast
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ["EOMT_AUDIT_OUTPUT"]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
ACTIVE = ROOT / "results/ablations/coco-val5000-mask-full"
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "offline"
os.environ["GIT_OPTIONAL_LOCKS"] = "0"
sys.dont_write_bytecode = True


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def local_path(value):
    for prefix, target in (("/workspace/", ROOT), ("/data/", ROOT / "data"),
                           ("/checkpoints/", ROOT / "checkpoints"),
                           ("/results/", ROOT / "results")):
        if value.startswith(prefix):
            return target / value[len(prefix):]
    return Path(value)


def protected_files():
    paths = []
    for folder in ("upstream/eomt", "checkpoints", "results", "scripts", "tests", "docs", "report"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and ".git" not in path.parts:
                if path.is_relative_to(ROOT / "results/audits") or path.is_relative_to(ACTIVE):
                    continue
                paths.append(path)
    paths.extend(ROOT / p for p in ("compose.yaml", "data/coco/annotations/panoptic_val2017.json"))
    return sorted(set(paths))


def snapshot():
    started = now()
    entries = {p.relative_to(ROOT).as_posix(): {"sha256": sha(p), "size": p.stat().st_size}
               for p in protected_files()}
    write(OUT / "integrity_before.json", {"started_utc": started, "finished_utc": now(), "files": entries})
    print("Protected files hashed", len(entries), flush=True)


def integrity():
    before = read(OUT / "integrity_before.json")["files"]
    current = {p.relative_to(ROOT).as_posix(): p for p in protected_files()}
    changed = [key for key, entry in before.items()
               if key not in current or sha(current[key]) != entry["sha256"]]
    added = sorted(set(current) - set(before))
    # The three new requested audit documents are expected additions.
    audit_docs = {"docs/FINAL_AUDIT.md", "docs/FULL_MASKING_RESULTS.md", "docs/REPORT_FINALIZATION_NOTES.md"}
    unexpected = sorted(set(added) - audit_docs)
    git = subprocess.run(["git", "-C", str(ROOT / "upstream/eomt"), "status", "--porcelain"],
                         capture_output=True, text=True)
    write(OUT / "integrity_after.json", {"checked_utc": now(), "protected_existing_files": len(before),
          "changed_or_missing": changed, "added_files": added, "unexpected_additions": unexpected,
          "upstream_git_status_returncode": git.returncode, "upstream_git_status": git.stdout,
          "status": "passed" if not changed and not unexpected and git.returncode == 0 and not git.stdout else "failed"})
    print("Integrity changed", changed, "unexpected additions", unexpected, flush=True)


def tests():
    os.environ["EOMT_RESULTS_DIR"] = str(OUT / "test-fixtures")
    (OUT / "test-fixtures").mkdir()
    modules = {name: bool(importlib.util.find_spec(name))
               for name in ("numpy", "PIL", "torch", "timm", "lightning", "yaml")}
    with (OUT / "tests.log").open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    write(OUT / "tests.json", {"checked_utc": now(), "command": "python -B -m unittest discover -s tests -v",
          "execution": "equivalent unittest discovery through this driver; stdout/stderr retained",
          "python": sys.version, "modules": modules, "tests_run": result.testsRun,
          "errors": [{"test": str(t), "traceback": trace} for t, trace in result.errors],
          "failures": [{"test": str(t), "traceback": trace} for t, trace in result.failures],
          "skipped": [{"test": str(t), "reason": reason} for t, reason in result.skipped],
          "successful": result.wasSuccessful()})
    print("Tests run", result.testsRun, "errors", len(result.errors), "failures", len(result.failures),
          "skipped", len(result.skipped), flush=True)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Unexpected verifier source for guarded replacement: " + old)
    return source.replace(old, new)


def check_settings(metrics, manifest):
    import numpy as np
    experiment = metrics["experiment"]
    assert manifest["experiment"] == experiment
    effective = experiment["effective"]
    requested = experiment["requested"]
    assert effective["queries"] == requested["queries"]
    assert effective["query_blocks"] == requested["query_blocks"]
    assert effective["attention"] == requested["attention"]
    assert effective["prediction_stage"] == "final only" and effective["checkpoint_modified"] is False
    assert effective["backbone_blocks"] == 12
    assert effective["query_block_indices_zero_based"] == list(range(12 - requested["query_blocks"], 12))
    retained = list(range(200))
    if requested["queries"] == 100:
        retained = sorted(int(i) for i in np.random.default_rng(requested["query_seed"]).choice(retained, 100, replace=False))
    assert effective["retained_query_ids"] == retained
    mode = requested["attention"]
    assert effective["masked_branch"] == (mode != "off")
    assert effective["mask_probabilities"] == (None if mode == "off" else [int(mode == "full")] * 3)
    assert requested["queries"] == 200 or requested["query_blocks"] == 3 and mode == "off"
    assert experiment["selection"]["count"] == metrics["evaluated_size"]
    assert experiment["selection"]["seed"] == 0
    for key in ("image_ids", "checkpoint_sha256", "annotations_sha256", "config_sha256"):
        assert metrics[key] == manifest[key], key


def verify(label, relative_run):
    run = ROOT / relative_run
    output = OUT / label
    output.mkdir()
    started = now()
    with (output / "verification.log").open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                metrics = read(run / "metrics.json")
                assert metrics["status"] == "complete"
                completed_input_hashes = None
                if label == "mask-full-5000":
                    assert read(run / "experiment.json")["status"] == "complete"
                    completed_input_hashes = {p.relative_to(run).as_posix(): sha(p)
                                              for p in run.rglob("*") if p.is_file()}
                assert metrics["dataset_size"] == 5000
                assert metrics["device"] == "cpu" and metrics["precision"] == "float32"
                assert metrics["mask_threshold"] == metrics["overlap_threshold"] == .8
                assert metrics["source_commit_expected"] == "aa017c34afbe023de9462b17ae6067b52fef9867"
                assert metrics["scorer_revision"] == "7bb4655548f98f3fedc07bf37e9040a992b054b0"
                if label == "baseline-full":
                    script = run / "verify_and_rescore.py"
                    source = script.read_text(encoding="utf-8")
                    source = replace_once(source, "RUN = Path(__file__).resolve().parent", "RUN = AUDIT_RUN")
                    source = replace_once(source, "ROOT = RUN.parents[2]", "ROOT = AUDIT_ROOT")
                    old = '    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\\n", encoding="utf-8")'
                    source = replace_once(source, old, "    audit_write(path, value)")
                    scope = {"__file__": str(script), "__name__": "baseline_saved_audit",
                             "AUDIT_RUN": run, "AUDIT_ROOT": ROOT}
                else:
                    manifest = read(run / "experiment.json")
                    assert manifest["status"] == "complete"
                    check_settings(metrics, manifest)
                    if label == "mask-full-5000":
                        baseline = read(ROOT / "results/evaluation/coco-val5000/metrics.json")
                        for key in ("image_ids", "checkpoint_sha256", "config_sha256", "annotations_sha256",
                                    "source_commit_expected", "scorer_revision", "mask_threshold",
                                    "overlap_threshold", "device", "precision", "model_id"):
                            assert metrics[key] == baseline[key], key
                        columns = ("image_id", "input_sha256", "ground_truth_sha256")
                        assert [tuple(row[k] for k in columns) for row in metrics["per_image"]] == [
                            tuple(row[k] for k in columns) for row in baseline["per_image"]]
                    script = ROOT / "scripts/verify_ablation.py"
                    source = script.read_text(encoding="utf-8")
                    source = replace_once(source, 'sha(Path(metrics["checkpoint"]))', 'sha(local_path(metrics["checkpoint"]))')
                    source = replace_once(source, 'sha(Path(metrics["config"]))', 'sha(local_path(metrics["config"]))')
                    if label == "mask-full-5000":
                        assert metrics["evaluated_size"] == 5000
                        assert metrics["experiment"]["requested"] == {"queries": 200, "query_blocks": 3, "attention": "full", "query_seed": 0}
                        sample = "sorted(int(i) for i in np.random.default_rng(0).choice(sorted(images), 500, replace=False))"
                        source = replace_once(source, sample, "sorted(images)")
                        source, count = re.subn(r"\b500\b", "5000", source)
                        assert count == 5
                    else:
                        assert metrics["evaluated_size"] == 500
                    scope = {"__file__": str(script), "__name__": "ablation_saved_audit", "local_path": local_path}

                def audit_write(path, value):
                    assert path.parent.resolve() == run.resolve()
                    assert path.name in {"verification.json", "rescored_metrics.json", "prediction_manifest.json"}
                    write(output / path.name, value)

                # Guarded output redirection preserves every original run artifact.
                if label != "baseline-full":
                    old = '    with path.open("x", encoding="utf-8") as stream:\n        json.dump(value, stream, indent=2, allow_nan=False)\n        stream.write("\\n")'
                    source = replace_once(source, old, "    audit_write(path, value)")
                scope["audit_write"] = audit_write
                if label != "baseline-full":
                    insertion = '    class_rows = list(csv.DictReader((run / "per_class.csv").open(encoding="utf-8")))'
                    extra = '''    _, independent_classes = stat.pq_average(categories, isthing=None)
    for category, values in independent_classes.items():
        for key in ("pq", "sq", "rq"):
            close(float(values[key]), metrics["per_class"][str(category)][key])
'''
                    source = replace_once(source, insertion, extra + insertion)
                ast.parse(source)
                write(output / "adapter.json", {"original_verifier": str(script.relative_to(ROOT)),
                      "original_sha256": sha(script), "adapted_sha256": hashlib.sha256(source.encode()).hexdigest(),
                      "input_run": relative_run, "output": str(output.relative_to(ROOT)),
                      "changes": "isolated output writer; host mount-path translation for ablations; full-set ID/count adaptation for M1"})
                exec(compile(source, str(script), "exec"), scope)
                if label != "baseline-full":
                    scope["verify"](run)
                if completed_input_hashes is not None:
                    after = {p.relative_to(run).as_posix(): sha(p) for p in run.rglob("*") if p.is_file()}
                    assert completed_input_hashes == after
                    write(output / "input_artifact_integrity.json", {"status": "passed",
                          "original_run_unchanged": True, "files": after})
                write(output / "audit_status.json", {"status": "passed", "started_utc": started, "finished_utc": now()})
                print("Completed", label, flush=True)
            except Exception:
                traceback.print_exc()
                write(output / "audit_status.json", {"status": "failed", "started_utc": started,
                      "finished_utc": now(), "traceback": traceback.format_exc()})
    print(label, read(output / "audit_status.json")["status"], flush=True)


def status():
    files = list((ACTIVE / "panoptic_predictions").glob("*.png"))
    manifest = read(ACTIVE / "experiment.json")
    observed = datetime.now(timezone.utc)
    elapsed = (observed - datetime.fromisoformat(manifest["started_at_utc"])).total_seconds()
    record = {"observed_utc": observed.isoformat(), "manifest_status": manifest["status"],
              "started_utc": manifest["started_at_utc"], "prediction_pngs": len(files),
              "elapsed_seconds": elapsed, "metrics_exist": (ACTIVE / "metrics.json").exists(),
              "latest_png_utc": datetime.fromtimestamp(max(p.stat().st_mtime for p in files), timezone.utc).isoformat() if files else None,
              "remaining_seconds_linear_estimate": elapsed * (5000 - len(files)) / len(files) if files else None,
              "container_state": "unavailable; Docker named-pipe access denied by sandbox"}
    if record["metrics_exist"]:
        record["metrics_status"] = read(ACTIVE / "metrics.json").get("status")
    with (OUT / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("snapshot", "tests", "verify", "integrity", "status"))
    parser.add_argument("--label")
    parser.add_argument("--run")
    args = parser.parse_args()
    if args.action == "verify":
        verify(args.label, args.run)
    else:
        globals()[args.action]()
