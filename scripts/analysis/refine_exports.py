"""Refine only figure presentation from saved analysis/cases; no metrics rerun."""
import copy
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scripts.analysis.analyze_saved import ROOT, INPUT, VARIANTS, LABEL, load, sha, write, class_map, figure_save

OUT = Path(os.environ["EOMT_ANALYSIS_OUTPUT"]).resolve()


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    destination = OUT / "qualitative_refined"
    destination.mkdir(exist_ok=False)
    gt = load(ROOT / "data/coco/annotations/panoptic_val2017.json")
    images = {r["id"]: r for r in gt["images"]}
    targets = {r["image_id"]: r for r in gt["annotations"]}
    cats = {r["id"]: r for r in gt["categories"]}
    pred = {a: {r["image_id"]: r for r in load(INPUT / a / "predictions.json")["annotations"]}
            for a in ["baseline", *VARIANTS]}
    examples = copy.deepcopy(load(OUT / "qualitative_examples.json"))
    for case in examples["cases"]:
        image_id, arm = case["image_id"], case["arm"]
        with Image.open(ROOT / "data/coco/val2017" / images[image_id]["file_name"]) as source:
            rgb = np.asarray(source.convert("RGB"))
        target = targets[image_id]
        maps, ids = {}, {}
        maps["GT"], ids["GT"] = class_map(ROOT / "data/coco/annotations/panoptic_val2017" / target["file_name"], target["segments_info"], True)
        for name in ("baseline", arm):
            annotation = pred[name][image_id]
            maps[name], ids[name] = class_map(INPUT / name / "panoptic_predictions" / annotation["file_name"], annotation["segments_info"])
        present = sorted({int(v) for array in maps.values() for v in np.unique(array) if v})
        # Distinct colors for the categories actually present in this case.
        # They are held fixed across GT/B0/variant; legends specify each class.
        order = list(range(0, 20, 2)) + list(range(1, 20, 2))
        lut = np.zeros((max(cats) + 1, 3), dtype=np.uint8)
        lut[0] = [25, 30, 36]
        for i, category in enumerate(present):
            lut[category] = np.array(plt.get_cmap("tab20")(order[i % 20])[:3]) * 255

        def overlay(name):
            result = (.35 * rgb + .65 * lut[maps[name]]).astype(np.uint8)
            edge = np.zeros(maps[name].shape, dtype=bool)
            edge[1:] |= ids[name][1:] != ids[name][:-1]
            edge[:, 1:] |= ids[name][:, 1:] != ids[name][:, :-1]
            result[edge] = 255
            return result

        valid = maps["GT"] != 0
        before, after = maps["baseline"] == maps["GT"], maps[arm] == maps["GT"]
        change = np.full((*valid.shape, 3), [235, 238, 242], dtype=np.uint8)
        change[~valid] = [48, 53, 60]
        change[valid & ~before & after] = [0, 158, 115]
        change[valid & before & ~after] = [213, 94, 0]
        height = 3.6 if rgb.shape[1] / rgb.shape[0] > 1.2 else 5.3
        fig, axes = plt.subplots(1, 5, figsize=(16, height))
        for ax, array, title in zip(axes, [rgb, overlay("GT"), overlay("baseline"), overlay(arm), change],
                                   ["Input", "Ground truth", "B0: baseline", LABEL[arm], "Correctness changes"]):
            ax.imshow(array, interpolation="nearest"); ax.axis("off"); ax.set_title(title, fontsize=10, weight="bold")
        delta = case["delta_correct_percentage_points"]
        number = f"{delta:+.6f}" if abs(delta) < .01 else f"{delta:+.2f}"
        fig.suptitle(f"{LABEL[arm]} | {case['selection']} | COCO {image_id} | class-agreement change {number} pp",
                     fontsize=12, weight="bold", y=.985)
        fig.legend(handles=[Patch(facecolor=lut[c] / 255, label=cats[c]["name"]) for c in present],
                   loc="lower center", bbox_to_anchor=(.5, .087), ncol=min(6, len(present)), frameon=False, fontsize=8)
        note = (f"Teal: newly correct ({case['newly_correct_pixels']:,} pixels). Orange: newly wrong ({case['newly_wrong_pixels']:,}). "
                "Gray: correctness unchanged. Dark: ignored GT void/crowd.\n"
                "Category colors are shared within this case; white lines mark segments. Class agreement is a selection diagnostic, not PQ or instance matching.")
        fig.text(.5, .012, note, ha="center", va="bottom", fontsize=7.5)
        fig.subplots_adjust(left=.008, right=.992, top=.83, bottom=.27, wspace=.045)
        name = f"{arm}_{case['selection']}_{image_id}"
        figure_save(fig, destination, name, vector=False)
        case["figure"] = f"qualitative_refined/{name}.png"
        case["palette"] = {str(c): [int(v) for v in lut[c]] for c in present}
    write(OUT / "qualitative_refined_examples.json", examples)
    with (OUT / "per_class_changes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), layout="constrained")
    for ax, arm in zip(axes, VARIANTS):
        ranked = sorted([r for r in rows if r["arm"] == arm and r["included"] == "True" and int(r["gt_segments"]) >= 10],
                        key=lambda r: (float(r["delta_pq"]), int(r["category_id"])))
        selected = sorted(ranked[:7] + ranked[-5:], key=lambda r: float(r["delta_pq"]))
        ax.barh([f"{r['name']} (n={r['gt_segments']})" for r in selected], [float(r["delta_pq"]) for r in selected],
                color=["#0072B2" if float(r["delta_pq"]) > 0 else "#D55E00" for r in selected], height=.66)
        ax.set_xlim(-75, 25); ax.axvline(0, color="#334155", lw=.8)
        ax.grid(axis="x", color="#E5E7EB", lw=.6); ax.set_axisbelow(True)
        for name in ("left", "top", "right"):
            ax.spines[name].set_visible(False)
        ax.set_title(LABEL[arm], weight="bold"); ax.set_xlabel("PQ-point change from B0 (shared scale)")
    fig.suptitle("Supported class changes: same scale across all interventions (GT ≥10 segments)", fontsize=12, weight="bold")
    figure_save(fig, OUT / "figures", "per_class_extremes_shared_scale")
    original = load(OUT / "provenance.json")
    assert all(sha(ROOT / key) == digest for key, digest in original["source_sha256"].items())
    write(OUT / "figure_refinement_provenance.json", {
        "script_sha256": sha(Path(__file__)), "all_original_source_hashes_unchanged": True,
        "metrics_recomputed": False, "qualitative_cases_changed": False,
        "changes": "Case-distinct display palette; compact aspect-aware layout; shared-scale class chart",
        "case_manifest_sha256": sha(OUT / "qualitative_refined_examples.json")})
    print("Saved nine refined qualitative comparisons and shared-scale per-class chart; source data unchanged")


if __name__ == "__main__":
    main()
