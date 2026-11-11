# Experimental analysis: EoMT fixed-checkpoint sensitivities

[Published aggregate metrics](../publication/results/verified_metrics.csv) and [source fingerprints](../publication/results/provenance.json) accompany this analysis. Detailed exports, figures and original `results/` paths below refer to preserved local artifacts and are not distributed.

Prepared 2026-11-09 from the completed five-arm, matched 500-image experiment.
**These are fixed-checkpoint sensitivity experiments, not independently
retrained model variants.** No inference, official evaluation, rescoring or
training was rerun for this analysis. Existing aggregate/per-class results and
per-image timers were read; saved masks and ground truth were used only to
construct qualitative selection diagnostics and illustrations.

The largest effect is later query introduction, which loses **18.9355 PQ points**
for only a modest observed latency reduction. Removing half the learned queries
loses **7.2900 points**, disproportionately in stuff classes. Full masking loses
**1.3502 points** and is slower in the recorded timings. The zero-probability
control retains baseline predictions exactly while adding computational work.

## Evidence, scope and artifacts

Inputs are [the ablation records](ABLATIONS.md) and all five arms' `metrics.json`,
`per_class.csv`, `timings.csv`, `verification.json`, `rescored_metrics.json`,
prediction JSONs/PNGs, suite records and clock audit in
`results/ablations/coco-val500/`. All arms use the same 500 IDs,
source revision, official EoMT-S 640 2x checkpoint, 640-pixel preprocessing,
postprocessing thresholds, category mapping and reference-scoring protocol.
They have 131 active categories: 78 things and 53 stuff. Toaster and hair drier
are absent under the scorer's active-category rule. All five runs already
passed separate-process independent rescoring.

The final analysis artifacts are in
`results/analysis/eomt-val500-final`.
A local artifact guide lists the preferred figures. Charts have 300-dpi PNG, PDF and editable-text SVG exports; qualitative
figures have 300-dpi PNG and PDF exports. The analysis scripts and a preliminary
rendering attempt are retained under `results/analysis/`; the final directory
is the authoritative report artifact set.

## Aggregate quality and largest effects

All scores and differences below use metric points, not raw 0–1 JSON values.

| Arm | PQ | SQ | RQ | Things PQ | Stuff PQ | PQ difference |
|---|---:|---:|---:|---:|---:|---:|
| B0: baseline | 47.7820 | 81.6096 | 57.1361 | 50.8259 | 43.3024 | 0.0000 |
| Q100: retained 100 queries | 40.4920 | 74.1415 | 48.1849 | 48.5092 | 28.6931 | -7.2900 |
| L2: final two joint blocks | 28.8466 | 68.1019 | 34.3289 | 29.9015 | 27.2940 | -18.9355 |
| M1: full masking | 46.4318 | 80.3132 | 55.3781 | 50.1985 | 40.8885 | -1.3502 |
| M0: zero control | 47.7820 | 81.6096 | 57.1361 | 50.8259 | 43.3024 | 0.0000 |

The aggregate quality figure is retained locally and is not included in this publication.

Q100 reduces Things PQ by **2.3166** but Stuff PQ by **14.6093**. L2 reduces
Things PQ by **20.9243** and Stuff PQ by **16.0084**. M1 reduces these groups by
**0.6274** and **2.4139**, respectively. Thus query removal is especially damaging
to stuff coverage, whereas later query introduction damages both groups heavily.

Because the same classes are active in every arm, overall macro PQ is exactly
`(78 × Things PQ + 53 × Stuff PQ) / 131`. This allows a descriptive decomposition:

| Intervention | Things contribution to overall PQ change | Stuff contribution | Stuff share of total PQ loss |
|---|---:|---:|---:|
| Q100 | -1.3794 | -5.9106 | 81.08% |
| L2 | -12.4588 | -6.4767 | 34.20% |
| M1 | -0.3736 | -0.9766 | 72.33% |

The decomposition is arithmetic on saved scores, not a causal attribution to
specific objects. See `effect_decomposition.csv`
and `Things/Stuff figure`.

### Recognition, matching and abstention

| Arm | TP | FP | FN | Exported segments | Predicted void, all native pixels |
|---|---:|---:|---:|---:|---:|
| B0 | 2,979 | 910 | 2,479 | 4,214 | 16.82% |
| Q100 | 2,320 | 713 | 3,138 | 3,260 | 38.91% |
| L2 | 1,847 | 443 | 3,611 | 2,381 | 36.63% |
| M1 | 2,876 | 839 | 2,582 | 4,025 | 19.20% |
| M0 | 2,979 | 910 | 2,479 | 4,214 | 16.82% |

These counts come from the saved verification records. Void fractions divide
their recorded void pixels by **137,055,049 total native image pixels**. Exported
segments need not equal TP+FP because the reference scorer ignores certain
void/crowd-overlapping predictions.

Relative to B0, Q100 loses 659 true positives and adds 659 false negatives; L2
loses 1,132 true positives and adds 1,132 false negatives. Both also predict fewer
segments and much more void. This is consistent with lost coverage/recognition,
not a simple improvement inferred from their lower FP counts. M1 has a smaller
103-TP loss and 103-FN increase.

SQ and RQ both fall: Q100 changes SQ/RQ by **-7.4681/-8.9512** points; L2 by
**-13.5077/-22.8072**; M1 by **-1.2964/-1.7580**. RQ therefore provides direct
evidence of worse matching/recognition. SQ deterioration should not be described
solely as poorer boundaries: classes with no TP receive SQ=0, and Q100/L2 create
12/21 newly zero-TP classes that had at least one TP in B0. M1 creates two.
Also, dataset macro PQ is the mean of per-class products, **not** the product of
the displayed macro SQ and macro RQ, so their aggregate changes cannot simply
be multiplied to assign causal percentages.

## Per-class analysis

All 133 category records, including the two inactive categories, are retained
for every arm in
`per_class_changes.csv`.
Counts below concern the 131 active categories, with a 1e-10-point equality
tolerance to avoid classifying floating-point noise as change.

| Arm | Improved classes | Degraded classes | Unchanged classes |
|---|---:|---:|---:|
| Q100 | 32 | 97 | 2 |
| L2 | 7 | 122 | 2 |
| M1 | 46 | 83 | 2 |
| M0 | 0 | 0 | 131 |

The following examples require **at least ten non-crowd GT segments** in the
subset. Support is segment count, not image count or training-set frequency.
The complete export also retains rarer classes rather than silently excluding
them from the official averages.

| Intervention | Supported gain examples: PQ change (GT support) | Supported loss examples: PQ change (GT support) |
|---|---|---|
| Q100 | sandwich +9.0203 (13); backpack +5.2294 (45); parking meter +4.8236 (19) | sand -69.3880 (26); ceiling-merged -68.9274 (39); grass-merged -63.0275 (100) |
| L2 | pizza +4.9602 (17); flower +3.2505 (16); skateboard +2.2116 (15) | floor-wood -52.2813 (26); bed -48.7559 (20); motorcycle -46.2929 (24) |
| M1 | knife +15.6609 (15); vase +10.8540 (28); apple +6.3066 (23) | shelf -18.6510 (19); railroad -14.0397 (12); baseball bat -10.7865 (13) |

The class-change figure with a common scale is retained locally and is not included in this publication.

The broad L2 decline (122 classes) is not explained by one rare-class collapse.
Q100 has several severe stuff-class failures: sand, ceiling-merged and
grass-merged, among others, become zero-TP classes. This supports the observed
group-level loss while leaving the mechanism uncertain: the retained queries
were learned jointly with the removed queries, and query identities may be
specialized. Their fixed-weight removal does not measure the capacity of a
model trained with only 100 queries.

Full masking has a mixed class profile. Its improvements on knife/vase do not
offset its more numerous declines. Even the supported examples can have only
12–28 targets, so large per-class changes are descriptive and potentially
unstable; no significance test or confidence interval was computed. The
all-class heatmap
includes every active class and its support, with a common color scale.

## Latency from saved per-image durations

Every timing vector was aligned by the same 500 image IDs. No outliers were
trimmed. A ratio below one means the variant's saved prediction-call duration
was shorter than B0's for that image.

| Arm | Mean (s) | Median (s) | p95 (s) | Mean change vs B0 | Median paired ratio | Faster / slower images |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 1.644492 | 1.589670 | 1.972252 | 0.00% | 1.0000 | — |
| Q100 | 1.323319 | 1.320269 | 1.417721 | -19.53% | 0.8256 | 497 / 3 |
| L2 | 1.555264 | 1.538615 | 1.728076 | -5.43% | 0.9615 | 334 / 166 |
| M1 | 2.109822 | 2.090391 | 2.354513 | +28.30% | 1.3125 | 14 / 486 |
| M0 | 1.956957 | 1.924682 | 2.179926 | +19.00% | 1.1825 | 48 / 452 |

Mean change is `(mean_variant / mean_B0 - 1) × 100`; the median paired ratio is
`median(t_variant,image / t_B0,image)`, which is a different statistic.
Raw aligned timings and additional extrema/p99 values are in
`paired_image_timings.csv`
and `latency_summary.csv`.

The latency distribution figure is retained locally and is not included in this publication.

In these measurements, Q100 trades lower quality for shorter prediction calls.
L2 is slower and less accurate than Q100, and its latency reduction from B0 is
small despite the largest quality loss. B0 is more accurate and faster than M1;
M0 has identical predictions but extra work for intermediate heads and mask
construction. These are observed relationships, not hardware-independent speedup
claims. Runs were sequential in a fixed order, with no randomized repetitions,
CPU utilization trace, thermal control or matched wall-clock operating state.
Box whiskers show p5/p95 and boxes show IQR, not confidence intervals.
The `PQ/median-latency chart`
contains only the five measured points, without an estimated performance curve.

### Laptop sleep: counter intervals, wall time and CPU compute are different

| Arm | Saved evaluator interval (s) | UTC start-to-finish (s) | UTC minus counter (s) |
|---|---:|---:|---:|
| B0 | 898.393748 | 898.320789 | -0.072959 |
| Q100 | 726.471654 | 726.391412 | -0.080242 |
| L2 | 847.979255 | 866.782523 | +18.803268 |
| M1 | 1125.930518 | 1125.785708 | -0.144810 |
| M0 | 1046.849078 | 4145.089252 | +3098.240174 |

The clock comparison figure is retained locally and is not included in this publication.

The reported laptop sleep is consistent with the large M0 clock gap:
**17.45 minutes on the saved counter versus 69.08 minutes by UTC**, a difference
of **51.64 minutes**. The suite as a whole recorded 80.20 counter minutes versus
132.15 UTC minutes. The earlier log notes a roughly 3,053-second delayed polling
call. These are related observations, not identical timing scopes. The smaller
L2 discrepancy is also preserved. No measured duration was adjusted or removed.

The inference/evaluation timers use `time.perf_counter()`. This measures elapsed
duration around code, including scheduling and other overhead, **not pure CPU
compute seconds**. Python distinguishes it from `process_time()`, which measures
process CPU time. Ordinary thread sleep is included in `perf_counter`; host/VM
suspension is a different situation. [Python 3.11 time documentation](https://docs.python.org/3.11/library/time.html#time.perf_counter).

On Linux, `CLOCK_MONOTONIC` excludes system suspension, whereas suspend-aware
clocks differ. This explains how a Linux/WSL counter and host-aligned UTC can
diverge during laptop sleep, **if that clock behavior applied in the VM**.
[Linux clock documentation](https://man7.org/linux/man-pages/man3/clock_gettime.3.html).
The run did not save clock-implementation diagnostics or OS power-event records,
so the exact sleep boundaries and VM clock mechanism cannot be independently
reconstructed from these artifacts. The 3,098.24-second difference is a measured
clock gap, not a verified exact sleep duration or missing-compute estimate.

No per-image timer contains a 51-minute inference: M0's maximum saved duration
is **7.972428 seconds**. Its prediction-call sum is 978.478540 seconds; the larger
counter total includes loading/export/scoring, and UTC includes the disruption.
No CPU `process_time` was recorded, so actual CPU compute time cannot be recovered.
Use the saved medians/distributions as descriptive observations, and report the
UTC interval for elapsed completion time. Do not infer that M0's model needed
69 minutes of neural computation. See `clock_audit.csv`.

## Representative qualitative examples

All panels use saved native-resolution masks and original COCO images. Segment
IDs were decoded and joined with each annotation's category IDs. White lines
show segment boundaries; display colors are fixed across GT/B0/variant **within
each case**, with a category legend. They need not match colors across different
cases. The saved categories had no display colors, so the renderer supplies a
deterministic palette; it does not alter class assignments or boundaries.

Examples were selected transparently after inspecting the saved data. For each
nontrivial intervention, the renderer selects the largest positive/negative
change and the nearest median nonzero change among images having at least 50%
valid GT pixels and three non-crowd GT segments. Ties use image ID. The diagnostic
is change in **semantic class-agreement percentage points on non-crowd, non-void
GT pixels**. It ignores instance separation and is **not per-image PQ**, an IoU
metric or a new official benchmark evaluation. Gray means correctness unchanged,
which can include both methods being wrong. Teal/orange indicate newly
correct/newly wrong pixels. Dark denotes ignored GT pixels.

| Arm | Selection | COCO ID | Class-agreement change | Observed example |
|---|---|---:|---:|---|
| Q100 | Largest improvement | 147205 | +95.7805 pp | Broad dining-table assignment agrees with the coarse GT where B0 assigns food-other. This is annotation-dependent, not proof of visually better food boundaries. |
| Q100 | Largest degradation | 438876 | -97.3803 pp | Person and fence regions that B0 labels correctly are largely void after removing queries. |
| Q100 | Median changed case | 577976 | -13.5465 pp | The construction-scene building region loses coverage; some smaller regions change as well. |
| L2 | Largest improvement | 530457 | +37.3001 pp | Flowers are assigned the GT flower category rather than B0's potted-plant category; other regions still contain errors. |
| L2 | Largest degradation | 308753 | -94.8422 pp | Most of the GT rock background becomes void while the small bear remains detected. |
| L2 | Median changed case | 579158 | -8.0120 pp | The airplane remains, but coverage of truck/bridge regions is lost. |
| M1 | Largest improvement | 296284 | +46.6782 pp | Shelf regions gain coverage; substantial paper/background disagreement still remains. |
| M1 | Largest degradation | 127530 | -76.9955 pp | The wall background becomes void while the foreground person remains. |
| M1 | Median changed case | 330396 | -0.000581 pp | 186 pixels become correct and 187 become wrong. Some larger class/void changes are neutral in this diagnostic because both predictions disagree with GT. |

The selected improvement and degradation panels are retained locally:

The L2 flower-category improvement panel is retained locally and is not included in this publication.

The Q100 person/fence coverage-loss panel is retained locally and is not included in this publication.

All nine preferred figures are in
qualitative_refined,
with exact selection details, pixel counts, palette values and source-image
hashes in `qualitative_refined_examples.json`.
The full 1,500-row selection audit is
`qualitative_selection_diagnostics.csv`.

Across all 500 images, class agreement improves/degrades/is unchanged on
73/425/2 images for Q100, 56/442/2 for L2, and 252/247/1 for M1. These are
**diagnostic counts**, not image-level PQ wins. M1 can improve class agreement on
slightly more images while still reducing dataset PQ: the diagnostic ignores
instance matching, class-balanced averaging and the magnitudes of changes.
Extreme examples are deliberately not presented as typical; median changed-case
panels provide context. Even these median panels represent the filtered eligible
set, not a random qualitative sample.

## Reproducibility, preservation and limitations

Analysis was run locally using Python 3.13, NumPy 2.2.6 and Matplotlib 3.10.8;
these are analysis-tool versions, not the CPU Docker evaluation environment.
Original scripts are retained under `results/analysis/`. The following historical commands produced the saved artifacts:

```powershell
python results/analysis/analyze_saved.py --output-dir results/analysis/eomt-val500-final
python results/analysis/refine_exports.py
python results/analysis/finalize_analysis.py
```

These commands were executed for artifact production; the destinations now
exist, so do not rerun them onto the saved artifacts. The main script accepts a
fresh output directory. Maintained copies under `scripts/analysis/` use `EOMT_ANALYSIS_OUTPUT` for refinement/finalization. For a separate repeat, run the maintained module commands below from the project root with a new output directory. None imports a model or the COCO
scorer. An initial rendering pass stopped because categories lacked display
colors; it is marked preliminary and preserved. Palette/layout refinements and
the shared-scale class chart changed presentation only, not selection or metrics.

The local analysis `provenance.json` contains
before/after SHA-256 checks for every saved source-ablation file plus protected
checkpoint/prior result records. All matched after analysis and figure refinement.
`artifact_manifest.json`
lists generated-file hashes, sizes and PNG dimensions; PNGs decoded and PDF
headers passed checks. Figures were visually inspected for layout and meaning.

Interpretation is limited to one pretrained checkpoint, one 500-image sample,
one query subset, one execution order and the existing evaluation protocol.
No confidence intervals, training-seed variance or independent training outcomes
are available. Neither these results nor selected pictures establish what a
separately trained K=100, L2=2 or always-masked architecture could achieve. There
was no new inference, evaluation, rescoring, training, checkpoint modification,
upstream modification or overwrite of prior experimental results.

## Maintained analysis commands

These commands require the original matched-pilot artifacts at `results/ablations/coco-val500/` and the local COCO inputs. They do not reconstruct those artifacts from the compact published summaries.

```bash
python -m scripts.analysis.analyze_saved --output-dir results/analysis/new-analysis
```

Set `EOMT_ANALYSIS_OUTPUT` to `results/analysis/new-analysis`, using PowerShell `$env:EOMT_ANALYSIS_OUTPUT = "results/analysis/new-analysis"` or POSIX `export EOMT_ANALYSIS_OUTPUT=results/analysis/new-analysis`, before adding the remaining exports:

```bash
python -m scripts.analysis.refine_exports
python -m scripts.analysis.finalize_analysis
```

Use a fresh analysis directory and do not point these commands at historical exports.
