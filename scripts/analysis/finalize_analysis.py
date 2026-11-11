"""Export final clock figure and descriptive effect decomposition from saved data."""
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from scripts.analysis.analyze_saved import ROOT, INPUT, ARMS, LABEL, load, sha, write, csv_write, figure_save

OUT = Path(os.environ["EOMT_ANALYSIS_OUTPUT"]).resolve()


def main():
    data = load(OUT / "analysis.json")
    metrics = {r["arm"]: r for r in data["metrics"]}
    base = load(INPUT / "baseline/metrics.json")
    things_weight = base["metrics"]["Things"]["n"] / base["metrics"]["All"]["n"]
    stuff_weight = base["metrics"]["Stuff"]["n"] / base["metrics"]["All"]["n"]
    decomposition, zero_tp = [], {}
    for arm in ARMS[1:]:
        row = metrics[arm]
        thing = things_weight * row["delta_things_pq"]
        stuff = stuff_weight * row["delta_stuff_pq"]
        assert abs(thing + stuff - row["delta_pq"]) < 1e-10
        decomposition.append({"arm": arm, "things_contribution_pq_points": thing,
            "stuff_contribution_pq_points": stuff, "total_delta_pq_points": row["delta_pq"],
            "stuff_share_of_total_loss_percent": 100 * stuff / row["delta_pq"] if row["delta_pq"] else None})
        after = load(INPUT / arm / "metrics.json")["per_class"]
        zero_tp[arm] = [{"category_id": int(key), "name": before["name"],
                         "gt_segments": before["tp"] + before["fn"], "baseline_tp": before["tp"]}
                        for key, before in base["per_class"].items() if before["tp"] > 0 and after[key]["tp"] == 0]
    csv_write(OUT / "effect_decomposition.csv", decomposition)
    write(OUT / "additional_findings.json", {"macro_pq_group_weights": {"things": things_weight, "stuff": stuff_weight},
          "effect_decomposition": decomposition, "new_zero_true_positive_classes": zero_tp})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(9.5, 4.1), layout="constrained")
    clocks = data["clocks"]
    y = np.arange(5)
    counter = [r["evaluator_seconds"] / 60 for r in clocks]
    utc = [r["utc_elapsed_seconds"] / 60 for r in clocks]
    ax.barh(y - .17, counter, height=.30, label="Recorded performance-counter interval", color="#4E5968")
    ax.barh(y + .17, utc, height=.30, label="UTC start-to-finish interval", color="#E69F00")
    ax.set_yticks(y, [LABEL[a] for a in ARMS]); ax.invert_yaxis(); ax.set_xlim(0, 80)
    ax.set_axisbelow(True); ax.grid(axis="x", color="#E5E7EB", lw=.6)
    ax.set_xlabel("Evaluation interval (minutes; independent verification excluded)")
    for i, (a, b) in enumerate(zip(counter, utc)):
        ax.text(a + .5, i - .17, f"{a:.2f}", va="center", fontsize=8)
        ax.text(b + .5, i + .17, f"{b:.2f}", va="center", fontsize=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Laptop-sleep disruption: counter interval and UTC elapsed time differ", weight="bold")
    figure_save(fig, OUT / "figures", "clock_intervals_final")
    provenance = load(OUT / "provenance.json")
    assert all(sha(ROOT / path) == digest for path, digest in provenance["source_sha256"].items())
    artifacts = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        row = {"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha(path)}
        if path.suffix == ".png":
            with Image.open(path) as image:
                image.load()
                row["dimensions"] = list(image.size)
                assert min(image.size) > 100
        if path.suffix == ".pdf":
            with path.open("rb") as stream:
                assert stream.read(5) == b"%PDF-"
        artifacts.append(row)
    write(OUT / "artifact_manifest.json", {"files": artifacts, "source_data_hashes_unchanged": True,
          "finalizer_script_sha256": sha(Path(__file__)), "no_inference_evaluation_or_training": True})
    print(f"Validated {len(artifacts)} generated artifacts; protected source hashes unchanged")


if __name__ == "__main__":
    main()
