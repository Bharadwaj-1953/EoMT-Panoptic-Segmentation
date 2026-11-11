"""Compare only runs that passed the separate saved-mask audit."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ["EOMT_AUDIT_OUTPUT"]).resolve()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verified(label):
    assert read(OUT / label / "audit_status.json")["status"] == "passed"
    report = read(OUT / label / "verification.json")
    assert report["status"] == "passed" and report["independent_rescore_matches"]
    return report


full_b0 = verified("baseline-full")
full_m1 = verified("mask-full-5000")
pilot_b0 = verified("pilot-baseline")
pilot_m1 = verified("pilot-mask-full")
assert full_b0["unique_images"] == full_m1["unique_images"] == 5000
assert pilot_b0["unique_images"] == pilot_m1["unique_images"] == 500
comparison = {}
rows = []
for population, baseline, variant in (("full_validation", full_b0, full_m1),
                                      ("matched_pilot", pilot_b0, pilot_m1)):
    differences = {}
    for group in ("All", "Things", "Stuff"):
        differences[group] = {}
        for metric in ("pq", "sq", "rq"):
            b0 = 100 * baseline["metrics"][group][metric]
            m1 = 100 * variant["metrics"][group][metric]
            delta = m1 - b0
            differences[group][metric] = delta
            rows.append({"population": population, "group": group, "metric": metric,
                         "baseline_points": b0, "M1_points": m1, "M1_minus_baseline_points": delta})
    comparison[population] = {"baseline": baseline["metrics"], "M1": variant["metrics"],
                              "delta_points": differences, "images": baseline["unique_images"]}
comparison["full_minus_pilot_effect_points"] = {
    group: {metric: comparison["full_validation"]["delta_points"][group][metric]
            - comparison["matched_pilot"]["delta_points"][group][metric]
            for metric in ("pq", "sq", "rq")}
    for group in ("All", "Things", "Stuff")}
metrics = read(ROOT / "results/ablations/coco-val5000-mask-full/metrics.json")
utc_seconds = (datetime.fromisoformat(metrics["finished_at_utc"])
               - datetime.fromisoformat(metrics["started_at_utc"])).total_seconds()
comparison["runtime"] = {"started_at_utc": metrics["started_at_utc"],
                         "finished_at_utc": metrics["finished_at_utc"],
                         "evaluator_seconds": metrics["total_evaluation_seconds"],
                         "utc_seconds": utc_seconds,
                         "inference_sum_seconds": metrics["inference_seconds"],
                         "mean_prediction_seconds": metrics["inference_seconds"] / 5000,
                         "independent_verification_seconds": full_m1["total_verification_seconds"],
                         "independent_rescore_seconds": full_m1["rescore_seconds"]}
comparison["created_at_utc"] = datetime.now(timezone.utc).isoformat()
comparison["metric_scale"] = "stored baseline/M1 metrics are 0..1; delta_points and CSV values are 0..100 points"
comparison["interpretation"] = "overlapping validation populations; one pretrained checkpoint; no training replication or significance claim"
with (OUT / "masking_comparison.json").open("x", encoding="utf-8") as stream:
    json.dump(comparison, stream, indent=2, allow_nan=False)
with (OUT / "masking_comparison.csv").open("x", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    if row["group"] == "All" or row["metric"] == "pq":
        print(row["population"], row["group"], row["metric"],
              *(f"{row[key]:.4f}" for key in ("baseline_points", "M1_points", "M1_minus_baseline_points")))
print("runtime", comparison["runtime"])
