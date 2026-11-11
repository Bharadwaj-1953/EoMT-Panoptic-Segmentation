"""Descriptive analysis/figures from saved EoMT outputs. No model/scorer imports."""
import argparse
import csv
import colorsys
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import textwrap

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/ablations/coco-val500"
ARMS = ["baseline", "queries100", "blocks2", "mask-full", "mask-zero"]
VARIANTS = ARMS[1:4]
LABEL = dict(zip(ARMS, ["B0: baseline", "Q100: 100 queries", "L2: 2 joint blocks", "M1: full masking", "M0: zero control"]))
COLOR = dict(zip(ARMS, ["#4E5968", "#0072B2", "#D55E00", "#009E73", "#CC79A7"]))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def csv_write(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure_save(fig, out, name, vector=True):
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    if vector:
        fig.savefig(out / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def class_map(path, segments, exclude_crowd=False):
    with Image.open(path) as image:
        rgb = np.asarray(image, dtype=np.uint32)
    ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    mapping = {int(s["id"]): int(s["category_id"]) for s in segments
               if not (exclude_crowd and s.get("iscrowd", 0))}
    if int(ids.max()) < 10000:
        lut = np.zeros(int(ids.max()) + 1, dtype=np.uint16)
        for key, value in mapping.items():
            if key < len(lut):
                lut[key] = value
        result = lut[ids]
    else:
        unique, inverse = np.unique(ids, return_inverse=True)
        values = np.array([mapping.get(int(key), 0) for key in unique], dtype=np.uint16)
        result = values[inverse].reshape(ids.shape)
    return result, ids


def plots(out, metrics, timings, classes, latency, clocks):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False, "axes.axisbelow": True,
        "axes.edgecolor": "#AEB5BD", "grid.color": "#E4E7EB", "grid.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    names = [LABEL[a] for a in ARMS]
    colors = [COLOR[a] for a in ARMS]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for ax, key in zip(axes, ("pq", "sq", "rq")):
        values = [100 * metrics[a]["metrics"]["All"][key] for a in ARMS]
        ax.barh(names, values, color=colors, height=.62)
        ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Metric points")
        ax.set_title(key.upper()); ax.grid(axis="x")
        for y, value in enumerate(values):
            ax.text(value + 1, y, f"{value:.2f}", va="center", fontsize=8)
    fig.suptitle("Fixed-checkpoint sensitivity on 500 matched COCO validation images", fontsize=13, weight="bold")
    figure_save(fig, out, "metrics_overview")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")
    for ax, group in zip(axes, ("Things", "Stuff")):
        values = [100 * metrics[a]["metrics"][group]["pq"] for a in ARMS]
        ax.barh(names, values, color=colors, height=.62); ax.invert_yaxis()
        ax.set_xlim(0, 65); ax.set_xlabel("Panoptic Quality (points)"); ax.set_title(f"{group} PQ")
        ax.grid(axis="x")
        for y, value in enumerate(values):
            ax.text(value + .7, y, f"{value:.2f}", va="center", fontsize=8)
    fig.suptitle("Query removal affects stuff more; later insertion affects both groups", fontsize=12, weight="bold")
    figure_save(fig, out, "things_stuff_pq")

    active = sorted([row for row in classes if row["arm"] == "queries100" and row["included"]],
                    key=lambda row: (not row["isthing"], row["category_id"]))
    fig, ax = plt.subplots(figsize=(8.1, 25), layout="constrained")
    values = np.array([[next(r["delta_pq"] for r in classes if r["category_id"] == row["category_id"] and r["arm"] == arm)
                        for arm in ARMS[1:]] for row in active])
    im = ax.imshow(values, aspect="auto", cmap="RdBu", vmin=-100, vmax=100, interpolation="nearest")
    ax.set_xticks(range(4), ["Q100", "L2", "M1", "M0"])
    ax.set_yticks(range(len(active)), [f"{r['name']}  (GT {r['gt_segments']})" for r in active], fontsize=6.4)
    ax.set_title("Per-class PQ change from baseline\n131 active classes; GT = non-crowd target segments", pad=13)
    fig.colorbar(im, ax=ax, shrink=.36, label="PQ-point difference (variant minus B0)")
    figure_save(fig, out, "per_class_heatmap")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), layout="constrained")
    for ax, arm in zip(axes, VARIANTS):
        supported = [r for r in classes if r["arm"] == arm and r["included"] and r["gt_segments"] >= 10]
        ranked = sorted(supported, key=lambda r: (r["delta_pq"], r["category_id"]))
        chosen = sorted(ranked[:7] + [r for r in ranked[-5:] if r not in ranked[:7]], key=lambda r: r["delta_pq"])
        ax.barh([f"{r['name']} (n={r['gt_segments']})" for r in chosen], [r["delta_pq"] for r in chosen],
                color=["#0072B2" if r["delta_pq"] > 0 else "#D55E00" for r in chosen], height=.67)
        ax.axvline(0, color="#334155", lw=.8); ax.grid(axis="x")
        ax.set_title(LABEL[arm]); ax.set_xlabel("PQ-point change from B0")
    fig.suptitle("Largest supported losses and best changes (GT support ≥10 segments)", fontsize=12, weight="bold")
    figure_save(fig, out, "per_class_extremes")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    for ax, arrays, labels, cs in (
        (axes[0], [timings[a] for a in ARMS], names, colors),
        (axes[1], [timings[a] / timings["baseline"] for a in ARMS[1:]], names[1:], colors[1:])):
        boxes = ax.boxplot(arrays, orientation="horizontal", tick_labels=labels, patch_artist=True,
                          whis=(5, 95), showfliers=True, widths=.53,
                          medianprops={"color": "black", "linewidth": 1.2},
                          flierprops={"marker": ".", "markersize": 3, "alpha": .45})
        for box, color in zip(boxes["boxes"], cs):
            box.set_facecolor(color); box.set_alpha(.65)
        ax.invert_yaxis(); ax.grid(axis="x"); ax.set_xscale("log")
    axes[0].set_xlabel("Saved prediction-call duration (s, log scale)")
    axes[1].axvline(1, color="#334155", linestyle="--", lw=1)
    axes[1].set_xlabel("Same-image duration / baseline duration (log scale)")
    axes[0].set_title("All 500 saved timings per arm")
    axes[1].set_title("Paired by image ID; 1 = baseline")
    fig.suptitle("Observed CPU latency — median / IQR, whiskers at p5 and p95; all outliers retained", fontsize=11, weight="bold")
    figure_save(fig, out, "latency_distributions")

    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
    offsets = {"baseline": (8, 8), "queries100": (-4, 10), "blocks2": (8, -18),
               "mask-full": (8, -18), "mask-zero": (8, 9)}
    for row in latency:
        arm = row["arm"]
        pq = 100 * metrics[arm]["metrics"]["All"]["pq"]
        ax.scatter(row["median_seconds"], pq, color=COLOR[arm], s=75, edgecolor="white", linewidth=.8, zorder=3)
        ax.annotate(LABEL[arm], (row["median_seconds"], pq), xytext=offsets[arm], textcoords="offset points", fontsize=8)
    ax.set(xlim=(0, 2.65), ylim=(0, 56), xlabel="Median saved prediction-call duration (s)", ylabel="PQ (points)")
    ax.grid(); ax.set_title("Accuracy versus observed median latency\nFixed checkpoint; timing is not a controlled benchmark")
    figure_save(fig, out, "pq_latency")

    fig, ax = plt.subplots(figsize=(9.5, 4), layout="constrained")
    y = np.arange(5)
    timer = [r["evaluator_seconds"] / 60 for r in clocks]
    wall = [r["utc_elapsed_seconds"] / 60 for r in clocks]
    ax.barh(y - .17, timer, height=.30, label="Saved performance-counter interval", color="#4E5968")
    ax.barh(y + .17, wall, height=.30, label="UTC start-to-finish interval", color="#E69F00")
    ax.set_yticks(y, names); ax.invert_yaxis(); ax.set_xlim(0, 80); ax.grid(axis="x")
    ax.set_xlabel("Evaluation interval (minutes; verification excluded)")
    for i, (a, b) in enumerate(zip(timer, wall)):
        ax.text(a + .5, i - .17, f"{a:.2f}", va="center", fontsize=8)
        ax.text(b + .5, i + .17, f"{b:.2f}", va="center", fontsize=8)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title("Laptop-sleep timing anomaly: the counter interval is not elapsed wall time")
    figure_save(fig, out, "clock_intervals")


def render_case(out, case, images, gt_annotations, prediction_index, categories):
    image_id, arm = case["image_id"], case["arm"]
    image_record = images[image_id]
    gt_ann = gt_annotations[image_id]
    source_path = ROOT / "data/coco/val2017" / image_record["file_name"]
    with Image.open(source_path) as source:
        source_rgb = np.asarray(source.convert("RGB"))
    maps, ids = {}, {}
    maps["GT"], ids["GT"] = class_map(ROOT / "data/coco/annotations/panoptic_val2017" / gt_ann["file_name"], gt_ann["segments_info"], True)
    for name in ("baseline", arm):
        annotation = prediction_index[name][image_id]
        maps[name], ids[name] = class_map(INPUT / name / "panoptic_predictions" / annotation["file_name"], annotation["segments_info"])
    valid = maps["GT"] != 0
    before, after = maps["baseline"] == maps["GT"], maps[arm] == maps["GT"]
    change = np.full((*valid.shape, 3), [235, 238, 242], dtype=np.uint8)
    change[~valid] = [48, 53, 60]
    change[valid & ~before & after] = [0, 158, 115]
    change[valid & before & ~after] = [213, 94, 0]
    color_lut = np.zeros((max(categories) + 1, 3), dtype=np.uint8)
    color_lut[0] = [25, 30, 36]
    for category, info in categories.items():
        # COCO's saved category records need not include display colors.
        # Deterministic category colors are visualization choices, not data.
        color_lut[category] = info.get("color", [round(255 * value) for value in
            colorsys.hsv_to_rgb((category * .618033988749895) % 1, .65, .90)])

    def overlay(name):
        result = (.35 * source_rgb + .65 * color_lut[maps[name]]).astype(np.uint8)
        boundaries = np.zeros(valid.shape, dtype=bool)
        boundaries[1:] |= ids[name][1:] != ids[name][:-1]
        boundaries[:, 1:] |= ids[name][:, 1:] != ids[name][:, :-1]
        result[boundaries] = [255, 255, 255]
        return result

    fig, axes = plt.subplots(1, 5, figsize=(16, 5.7))
    panels = [source_rgb, overlay("GT"), overlay("baseline"), overlay(arm), change]
    for ax, array, title in zip(axes, panels, ["Input", "Ground truth", "B0: baseline", LABEL[arm], "Semantic correctness changes"]):
        ax.imshow(array, interpolation="nearest"); ax.axis("off"); ax.set_title(title, fontsize=10)
    title = f"{LABEL[arm]} | {case['selection']} | COCO {image_id} | diagnostic change {case['delta_correct_percentage_points']:+.2f} pp"
    fig.suptitle(title, fontsize=12, weight="bold", y=.98)
    present = sorted(set(int(v) for key in maps for v in np.unique(maps[key]) if v != 0))
    handles = [Patch(facecolor=color_lut[c] / 255, label=categories[c]["name"]) for c in present]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .065), ncol=min(6, max(1, len(handles))),
               frameon=False, fontsize=8)
    note = (f"Teal = newly correct ({case['newly_correct_pixels']:,} pixels); orange = newly wrong ({case['newly_wrong_pixels']:,}); "
            "light gray = correctness unchanged; dark = ignored GT void/crowd.\n"
            "Same deterministic category colors across panels; white lines delineate segments. Diagnostic is class agreement, not PQ or instance matching.")
    fig.text(.5, .015, note, ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=.01, right=.99, top=.87, bottom=.25, wspace=.04)
    name = f"{arm}_{case['selection']}_{image_id}"
    figure_save(fig, out, name, vector=False)
    case["figure"] = f"qualitative/{name}.png"
    case["categories_displayed"] = {str(c): categories[c]["name"] for c in present}
    case["source_image_sha256"] = sha(source_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/analysis/eomt-val500")
    out = parser.parse_args().output_dir.resolve()
    if not out.is_relative_to(ROOT / "results/analysis") or out == ROOT / "results/analysis":
        raise ValueError("Output must be a new descendant of results/analysis")
    out.mkdir(parents=True, exist_ok=False)
    charts, qualitative = out / "figures", out / "qualitative"
    charts.mkdir(); qualitative.mkdir()
    source_files = sorted(p for p in INPUT.rglob("*") if p.is_file())
    protected = [ROOT / "checkpoints/coco_panoptic_eomt_small_640_2x/pytorch_model.bin",
                 ROOT / "results/evaluation/coco-val5000/metrics.json", ROOT / "results/ablations/smoke-test/summary.json"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_files + protected}
    metrics, times, verification, predictions = {}, {}, {}, {}
    for arm in ARMS:
        metrics[arm] = load(INPUT / arm / "metrics.json")
        verification[arm] = load(INPUT / arm / "verification.json")
        rescored = load(INPUT / arm / "rescored_metrics.json")
        with (INPUT / arm / "per_class.csv").open(encoding="utf-8") as stream:
            csv_classes = list(csv.DictReader(stream))
        assert len(csv_classes) == 133
        for row in csv_classes:
            for key in ("pq", "sq", "rq", "iou_sum", "tp", "fp", "fn"):
                assert np.isclose(float(row[key]), metrics[arm]["per_class"][row["category_id"]][key], rtol=0, atol=1e-12)
        with (INPUT / arm / "timings.csv").open(encoding="utf-8") as stream:
            timing_rows = list(csv.DictReader(stream))
        assert [int(r["image_id"]) for r in timing_rows] == metrics[arm]["image_ids"]
        assert len(timing_rows) == len(set(metrics[arm]["image_ids"])) == 500
        times[arm] = np.array([float(r["inference_seconds"]) for r in timing_rows])
        assert abs(float(times[arm].sum()) - metrics[arm]["inference_seconds"]) < 1e-8
        assert verification[arm]["status"] == "passed" and verification[arm]["independent_rescore_matches"]
        for group in ("All", "Things", "Stuff"):
            for key in ("pq", "sq", "rq"):
                assert abs(rescored["metrics"][group][key] - metrics[arm]["metrics"][group][key]) < 1e-12
        predictions[arm] = {r["image_id"]: r for r in load(INPUT / arm / "predictions.json")["annotations"]}
    expected = metrics["baseline"]["image_ids"]
    assert all(metrics[a]["image_ids"] == expected for a in ARMS)
    active = [int(k) for k, v in metrics["baseline"]["per_class"].items() if v["included"]]
    classes, per_class_summary, latency, clocks, overview = [], {}, [], [], []
    for arm in ARMS:
        m, base = metrics[arm], metrics["baseline"]
        overview.append({"arm": arm, **{k: 100 * m["metrics"]["All"][k] for k in ("pq", "sq", "rq")},
            "things_pq": 100 * m["metrics"]["Things"]["pq"], "stuff_pq": 100 * m["metrics"]["Stuff"]["pq"],
            **{f"delta_{k}": 100 * (m["metrics"]["All"][k] - base["metrics"]["All"][k]) for k in ("pq", "sq", "rq")},
            "delta_things_pq": 100 * (m["metrics"]["Things"]["pq"] - base["metrics"]["Things"]["pq"]),
            "delta_stuff_pq": 100 * (m["metrics"]["Stuff"]["pq"] - base["metrics"]["Stuff"]["pq"]),
            "tp": verification[arm]["tp_total"], "fp": verification[arm]["fp_total"], "fn": verification[arm]["fn_total"],
            "predicted_segments": verification[arm]["prediction_segments"], "void_pixels": verification[arm]["void_pixels"]})
        ratio = times[arm] / times["baseline"]
        latency.append({"arm": arm, "mean_seconds": float(times[arm].mean()), "median_seconds": float(np.median(times[arm])),
            "p95_seconds": float(np.percentile(times[arm], 95)), "p99_seconds": float(np.percentile(times[arm], 99)),
            "min_seconds": float(times[arm].min()), "max_seconds": float(times[arm].max()),
            "mean_latency_change_percent": float(100 * (times[arm].mean() / times["baseline"].mean() - 1)),
            "paired_median_ratio": float(np.median(ratio)), "faster_images": int((times[arm] < times["baseline"]).sum()),
            "slower_images": int((times[arm] > times["baseline"]).sum()), "inference_sum_seconds": float(times[arm].sum())})
        utc = (datetime.fromisoformat(m["finished_at_utc"]) - datetime.fromisoformat(m["started_at_utc"])).total_seconds()
        clocks.append({"arm": arm, "evaluator_seconds": m["total_evaluation_seconds"], "utc_elapsed_seconds": utc,
                       "utc_minus_counter_seconds": utc - m["total_evaluation_seconds"]})
        for key, before in base["per_class"].items():
            after = m["per_class"][key]
            assert before["included"] == after["included"]
            assert before["tp"] + before["fn"] == after["tp"] + after["fn"]
            classes.append({"arm": arm, "category_id": int(key), "name": before["name"], "isthing": bool(before["isthing"]),
                "included": before["included"], "gt_segments": before["tp"] + before["fn"],
                "baseline_pq": 100 * before["pq"], "variant_pq": 100 * after["pq"],
                **{f"delta_{k}": 100 * (after[k] - before[k]) for k in ("pq", "sq", "rq")},
                **{f"delta_{k}": after[k] - before[k] for k in ("tp", "fp", "fn")}})
        rows = [r for r in classes if r["arm"] == arm and r["included"]]
        supported = [r for r in rows if r["gt_segments"] >= 10]
        per_class_summary[arm] = {"improved": sum(r["delta_pq"] > 1e-10 for r in rows),
            "degraded": sum(r["delta_pq"] < -1e-10 for r in rows),
            "unchanged": sum(abs(r["delta_pq"]) <= 1e-10 for r in rows),
            "largest_gains": sorted(rows, key=lambda r: (-r["delta_pq"], r["category_id"]))[:10],
            "largest_losses": sorted(rows, key=lambda r: (r["delta_pq"], r["category_id"]))[:10],
            "supported_gains": sorted(supported, key=lambda r: (-r["delta_pq"], r["category_id"]))[:10],
            "supported_losses": sorted(supported, key=lambda r: (r["delta_pq"], r["category_id"]))[:10]}
    csv_write(out / "metrics_comparison.csv", overview)
    csv_write(out / "per_class_changes.csv", classes)
    csv_write(out / "latency_summary.csv", latency)
    csv_write(out / "clock_audit.csv", clocks)
    csv_write(out / "paired_image_timings.csv", [
        {"image_id": image_id, **{a + "_seconds": float(times[a][i]) for a in ARMS},
         **{a + "_ratio_to_b0": float(times[a][i] / times["baseline"][i]) for a in ARMS[1:]}}
        for i, image_id in enumerate(expected)])
    print("Metrics, per-class changes and paired timing summaries saved", flush=True)
    plots(charts, metrics, times, classes, latency, clocks)
    full = load(ROOT / "data/coco/annotations/panoptic_val2017.json")
    images = {r["id"]: r for r in full["images"]}
    annotations = {r["image_id"]: r for r in full["annotations"]}
    categories = {r["id"]: r for r in full["categories"]}
    diagnostics = []
    total_image_pixels = sum(images[i]["width"] * images[i]["height"] for i in expected)
    for index, image_id in enumerate(expected):
        target = annotations[image_id]
        truth, _ = class_map(ROOT / "data/coco/annotations/panoptic_val2017" / target["file_name"], target["segments_info"], True)
        valid = truth != 0
        base_ann = predictions["baseline"][image_id]
        baseline, _ = class_map(INPUT / "baseline/panoptic_predictions" / base_ann["file_name"], base_ann["segments_info"])
        before = baseline == truth
        for arm in VARIANTS:
            pred_ann = predictions[arm][image_id]
            variant, _ = class_map(INPUT / arm / "panoptic_predictions" / pred_ann["file_name"], pred_ann["segments_info"])
            after = variant == truth
            n = int(valid.sum())
            gained = int((valid & ~before & after).sum())
            lost = int((valid & before & ~after).sum())
            diagnostics.append({"arm": arm, "image_id": image_id, "valid_gt_pixels": n,
                "image_pixels": int(valid.size), "gt_noncrowd_segments": sum(not s["iscrowd"] for s in target["segments_info"]),
                "newly_correct_pixels": gained, "newly_wrong_pixels": lost,
                "baseline_correct_fraction": float((valid & before).sum() / n) if n else None,
                "variant_correct_fraction": float((valid & after).sum() / n) if n else None,
                "delta_correct_percentage_points": 100 * (gained - lost) / n if n else None})
        if (index + 1) % 100 == 0:
            print(f"Read saved prediction/GT masks for {index + 1}/500 images; no inference or scorer called", flush=True)
    csv_write(out / "qualitative_selection_diagnostics.csv", diagnostics)
    cases = []
    diagnostic_summary = {}
    for arm in VARIANTS:
        rows = [r for r in diagnostics if r["arm"] == arm and r["valid_gt_pixels"]]
        diagnostic_summary[arm] = {"improved_class_agreement_images": sum(r["delta_correct_percentage_points"] > 0 for r in rows),
                                  "degraded_class_agreement_images": sum(r["delta_correct_percentage_points"] < 0 for r in rows),
                                  "unchanged_class_agreement_images": sum(r["delta_correct_percentage_points"] == 0 for r in rows)}
        eligible = [r for r in rows if r["valid_gt_pixels"] >= .5 * r["image_pixels"] and r["gt_noncrowd_segments"] >= 3]
        positive = sorted([r for r in eligible if r["delta_correct_percentage_points"] > 0], key=lambda r: (-r["delta_correct_percentage_points"], r["image_id"]))
        negative = sorted([r for r in eligible if r["delta_correct_percentage_points"] < 0], key=lambda r: (r["delta_correct_percentage_points"], r["image_id"]))
        changed = [r for r in eligible if r["delta_correct_percentage_points"] != 0]
        choices = []
        if positive:
            choices.append((positive[0], "largest-improvement"))
        if negative:
            choices.append((negative[0], "largest-degradation"))
        if changed:
            median = float(np.median([r["delta_correct_percentage_points"] for r in changed]))
            choices.append((min(changed, key=lambda r: (abs(r["delta_correct_percentage_points"] - median), r["image_id"])), "median-changed-case"))
        for row, selection in choices:
            case = {**row, "selection": selection}
            render_case(qualitative, case, images, annotations, predictions, categories)
            cases.append(case)
    write(out / "qualitative_examples.json", {"selection_rule": "Among images with >=50% valid GT pixels and >=3 non-crowd GT segments, choose largest positive/negative class-agreement delta and nearest median nonzero delta; ties by image ID. Post-hoc diagnostic selection, not PQ.", "cases": cases})
    analysis = {"study_type": "fixed-checkpoint sensitivity, not independently retrained architectures",
        "image_count": 500, "active_categories": len(active), "metrics": overview,
        "per_class": per_class_summary, "latency": latency, "clocks": clocks,
        "diagnostic_image_counts": diagnostic_summary, "total_native_image_pixels": total_image_pixels,
        "void_pixel_fraction": {a: verification[a]["void_pixels"] / total_image_pixels for a in ARMS},
        "qualitative_cases": [{k: v for k, v in c.items() if k not in ("categories_displayed",)} for c in cases],
        "limitations": ["No new PQ evaluation, model inference, or training; existing metrics reused",
                        "Pixel agreement diagnostics ignore GT crowd/void and do not measure instance separation or PQ",
                        "One fixed subset, one query subset, no training-seed replicates or inferential significance tests",
                        "Performance-counter durations are not pure CPU compute time; sleep disruption limits runtime claims"]}
    write(out / "analysis.json", analysis)
    assert all(sha(ROOT / name) == digest for name, digest in hashes.items()), "Source artifact modified"
    write(out / "provenance.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script_sha256": sha(Path(__file__)), "runtime": {"python": platform.python_version(),
        "numpy": np.__version__, "matplotlib": matplotlib.__version__}, "source_sha256": hashes,
        "all_source_hashes_unchanged": True, "no_model_or_scorer_imports": True})
    (out / "README.md").write_text(
        "# Saved-output EoMT analysis\n\nFixed-checkpoint sensitivities, not retrained architectures.\n\n"
        "- `analysis.json`: measured/derived summaries.\n- `figures/`: seven charts, each PNG (300 dpi), PDF and SVG.\n"
        "- `qualitative/`: nine selected five-panel comparisons, PNG and PDF.\n"
        "- `per_class_changes.csv`: all 133 categories and all five arms, including support.\n"
        "- `paired_image_timings.csv`, `latency_summary.csv`, `clock_audit.csv`: untrimmed timing analysis.\n"
        "- `qualitative_selection_diagnostics.csv`, `qualitative_examples.json`: transparent post-hoc selection.\n"
        "- `provenance.json`: protected source hashes before/after.\n\n"
        "No inference, official evaluation, rescoring or training was run. Pixel-agreement maps are qualitative selection diagnostics, not PQ.\n",
        encoding="utf-8")
    print(f"Analysis complete: {out}; {len(cases)} qualitative panels; all source hashes unchanged", flush=True)


if __name__ == "__main__":
    main()
