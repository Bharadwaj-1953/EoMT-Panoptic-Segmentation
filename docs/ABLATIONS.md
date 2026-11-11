# Fixed-checkpoint sensitivity experiments

Paths under `results/` describe preserved local artifacts, which are not distributed. Published aggregate scores are available in [verified_metrics.csv](../publication/results/verified_metrics.csv); full and pilot masking differences are in [masking_comparison.csv](../publication/results/masking_comparison.csv).

Implemented 2026-11-09 from the preserved local ablation plan. **All five
real-checkpoint 500-image experiments have now completed and passed artifact
verification and independent saved-mask rescoring. No full-validation ablation
experiments or training have been run.** These experiments examine interventions on the authors'
segmentation-trained checkpoint, not independently trained architectural variants.

## Implemented arms

| Arm | CLI setting | Effective change |
|---|---|---|
| B0 | Defaults | Original 200-query, three-joint-block, unmasked inference |
| Q100 | `--queries 100` | Retain 100 seeded, uniformly selected learned embeddings before forward |
| L2 | `--query-blocks 2` | Insert queries for the final two encoder blocks; keep all backbone blocks |
| M0 | `--attention zero` | Enable intermediate heads and mask construction; force query-to-patch attention fully open |
| M1 | `--attention full` | Enable intermediate heads and retain every predicted query-to-patch restriction |

Only one factor may change per run. Defaults remain 200 queries, three joint
blocks and masking off. Unsupported counts, combined interventions and repeated
application to the same model fail explicitly. The separate
[ablation entry point](../scripts/evaluate_ablations.py) loads a fresh original
model via the unchanged [baseline loader](../scripts/infer_panoptic.py) and then
applies [instance-local interventions](../scripts/ablation_interventions.py).
No checkpoint is saved or rewritten. The original baseline CLI still uses its
original first-N/all-image selection and unmodified prediction path.

Q100 replaces the in-memory embedding with exactly the selected checkpoint rows
and updates both the embedding count and `num_q`. Query IDs are sampled with
`numpy.default_rng(query_seed)`, then sorted into their original order. It does
not add random queries or discard predictions after a 200-query forward. L2
changes `num_blocks` after the complete original checkpoint load, preserving
all weights and the inactive three-element probability buffer.

M0/M1 explicitly overwrite all three in-memory mask probabilities after loading.
An instance-bound `_disable_attn_mask` replacement handles only probability
endpoints 0/1 in evaluation mode. This deliberately makes M0 exactly unrestricted
without random draws, avoiding upstream's rare `rand == 0` edge case. All other
upstream mask construction, attention and postprocessing remain in use. M0 still
computes intermediate heads/masks; it is an overhead control rather than the
fast unmasked path. Every arm exports only final-stage predictions. No upstream
class or source file is monkey-patched globally.

## Individual-arm execution commands

These commands specify the settings used in the measured suite below. The suite
runner launched each arm in a fresh process and then independently verified it.
The displayed output directories now exist: use a new parent for any repeat;
existing directories are deliberately rejected.
Run from the project root with the existing CPU image. Compose disables networking
and mounts upstream, checkpoints and input data read-only. No rebuild or download
is needed because scripts and tests are bind-mounted.

All five commands use the same 500 images: a uniform sample without replacement
from sorted validation IDs using sample seed 0, followed by sorting the selected
IDs. JSON input order and experiment settings do not affect selection. Each
manifest records the IDs and NumPy version, so compare ID lists before comparing
scores. The query-removal seed is separate from the image-sampling seed.

```powershell
# B0: fresh matched-subset baseline and timing control
docker compose run --rm --no-deps --pull never cpu python /workspace/scripts/evaluate_ablations.py --sample-size 500 --sample-seed 0 --threads 4 --output-dir results/ablations/coco-val500/baseline

# Q100: remove half of the learned queries, selected without validation feedback
docker compose run --rm --no-deps --pull never cpu python /workspace/scripts/evaluate_ablations.py --queries 100 --query-seed 0 --sample-size 500 --sample-seed 0 --threads 4 --output-dir results/ablations/coco-val500/queries100

# L2: later query introduction, retaining all encoder blocks
docker compose run --rm --no-deps --pull never cpu python /workspace/scripts/evaluate_ablations.py --query-blocks 2 --sample-size 500 --sample-seed 0 --threads 4 --output-dir results/ablations/coco-val500/blocks2

# M0: explicit zero-probability control
docker compose run --rm --no-deps --pull never cpu python /workspace/scripts/evaluate_ablations.py --attention zero --sample-size 500 --sample-seed 0 --threads 4 --output-dir results/ablations/coco-val500/mask-zero

# M1: deterministic reactivation of predicted attention restrictions
docker compose run --rm --no-deps --pull never cpu python /workspace/scripts/evaluate_ablations.py --attention full --sample-size 500 --sample-seed 0 --threads 4 --output-dir results/ablations/coco-val500/mask-full
```

For a later small check, use `--sample-size 10` and new directory names for B0
and M0. This is a **seeded ten-image sample**, not the old evaluator's first ten
IDs. Always compare arms on the same ID lists. The plan needs only this small
M0 control initially; its 500-image command above is available if expanded.
`--sample-size 5000` selects every ID from the current 5,000-image dataset, but
no full ablation was authorized or executed in this phase.

Optional path arguments are `--images`, `--panoptic`, `--annotations` and
`--checkpoint`, with the same local defaults as baseline evaluation. The workflow
supports CPU float32 only. Sample size must be positive and no greater than the
dataset size; it is never silently clipped. `--threads` sets intra-op threads;
actual inter-op threads are recorded without changing an already initialized
thread pool. Use a fresh process for every arm. Timing comparisons must retain
identical device, precision and threading.

## Output isolation and provenance

The shared COCO evaluator accepts an explicit selector and experiment metadata
only when called by the new runner. Its reference scorer, category conversion,
resize/pad, thresholds, original-resolution predictions, and dataset-level PQ
accumulation are reused. Ablation output must be a **new descendant** of
`results/ablations/`; the root itself, path escapes, existing directories and
`results/evaluation/...` are rejected. Atomic directory creation also rejects
concurrent attempts to use the same run path. Existing baseline artifacts are
not opened for writing. Choose another directory for a retry; there is no resume
or overwrite option.

| Artifact | Contents |
|---|---|
| `experiment.json` | Requested/effective interventions, retained query IDs, zero-based joint-block indices, explicit active mask probabilities, image-selection algorithm/seed/IDs, source hashes, checkpoint/config/annotation hashes, runtime and timing |
| `metrics.json` | All standard baseline evaluation fields plus the same nested experiment record and UTC start/finish timestamps |
| `predictions.json`, `panoptic_predictions/*.png` | Final-stage COCO annotations and RGB-encoded segment IDs |
| `ground_truth_subset.json` | Selected native-resolution target metadata with original categories/crowd flags |
| `metrics.csv`, `per_class.csv`, `timings.csv` | Group PQ/SQ/RQ, per-class scores/IoU/TP/FP/FN and per-image prediction durations/input/GT hashes |

`experiment.json` is initially marked `running` after successful model loading
and directory creation; it becomes `complete` after metrics are written. An
interrupted run can retain partial files and `running` status: do not treat it
as a completed result. Failures before output creation leave no run artifact.
The retained parent checkpoint is identified by SHA-256; source commit is the
audited expected revision, while actual wrapper/config content hashes are saved.
Runtime records Python, OS/platform, PyTorch, NumPy, intra-/inter-op thread counts
and the process argument vector. The argument vector is the Python invocation,
not a reconstructed Docker host command.

`model_load_seconds` includes original loading plus intervention setup;
`intervention_seconds` separately measures the setup. `inference_seconds` sums
per-image prediction calls (preprocessing, forward, restoration and conversion),
excluding PNG export and scoring. Total time includes model setup, data checks,
export and scoring, and is measured before final metadata serialization. No GPU
memory or separate forward-only profiler has been implemented.

Raw metric values are on 0–1. Multiply by 100 for PQ points. Compare pilot PQ
only against B0 on the same sampled IDs, not against the full-set 46.7364 PQ.
Seeded query removal measures a particular set of learned queries; it does not
measure a separately trained K=100 model. Later query insertion and reactivated
masking likewise measure fixed-weight sensitivity, not architectural optimality
or the benefit of training-time annealing.

## Automated validation

Executed in the existing CPU Docker image on 2026-11-09:

```powershell
docker compose run --rm --no-deps --pull never --workdir /workspace cpu python -m unittest discover -s tests -v
```

**Result: 31 tests passed, zero skipped, 5.514 seconds of test-run time.**
Nine new tests in [test_ablations.py](../tests/test_ablations.py) cover:

- Exact final-logit and weight equivalence when interventions are disabled.
- 100 retained checkpoint query rows, matching output counts, and deterministic
  query selection with differing seeds producing differing subsets.
- Actual token sequence lengths entering all 12 encoder blocks, confirming query
  insertion in blocks 10/11 for L2 (zero-based), without deleting encoder blocks.
- Explicit endpoint probability overrides, deterministic fully open M0 masks,
  query-to-patch-only M1 restrictions, four prediction stages and finite outputs.
- M0 final logits matching B0 within `rtol=1e-5`, `atol=1e-6` on the tiny synthetic
  model, with no RNG consumption by endpoint masking.
- Exact reproducibility of the planned 500-of-5,000 ID sampler using generated
  integer IDs; no real COCO files are read for this test.
- Synthetic end-to-end COCO export/scoring for all five arms, provenance fields,
  byte-preserved fixture checkpoint, separate outputs and overwrite/path rejection.
- Rejection of invalid settings, combined factors, training mode and repeated
  application to an already modified model.

The model tests execute real upstream EoMT and timm attention on a randomly
initialized, tiny 32-pixel, 16-channel, 12-block backbone. Workflow tests use
generated 2 x 2 images and predictor doubles, never pretrained inference. The
existing 22 inference/evaluation tests also passed. Matplotlib/Pyparsing emitted
non-fatal deprecation warnings.

The real checkpoint SHA-256 was checked read-only after testing and remains
`7365b41a8a129afc74c12e5dd7af534146286d02822e4b89b7bbe8ea613ff74d`.
Upstream remained clean at `aa017c34afbe023de9462b17ae6067b52fef9867`.

## Limitations and next-phase checks

All five configurations have now been exercised with the real pretrained weights
on the same 500-image subset, in addition to the earlier single-image smoke test.
The zero control agrees with baseline on all saved predictions in this subset;
this does not establish agreement on every image or platform. GPU execution,
controlled latency/memory benchmarks, paired bootstrap intervals,
cached-baseline subset scoring and comparison plots are not implemented here.
The shared scorer remains the official COCO reference implementation, whose
distinction from upstream's custom metric is documented in
[PAPER_COMPARISON.md](PAPER_COMPARISON.md).

## Real-checkpoint single-image smoke test

Completed 2026-11-09 in CPU Docker, without rebuilding or downloading anything:

```powershell
docker compose run --rm --no-deps --pull never cpu python /workspace/scripts/smoke_ablations.py
```

The new [smoke runner](../scripts/smoke_ablations.py) selects the first sorted ID
from the **500-image seed-0 subset**, not a fresh sample of size one. It executed
only image **1353**, `000000001353.jpg`, with native dimensions **375 wide x 500
high**. It loaded the official EoMT-S 640 2x checkpoint afresh for each arm and
for a separate unmodified reference: six single-image forward passes, all on
the same image. Computing the subset ID manifest did not run inference on the
other 499 images. No PQ evaluation or training was performed.

| Configuration | Status | Prediction seconds | Segments | Semantic pixels different from B0 |
|---|---|---:|---:|---:|
| Baseline B0 | Passed | 1.912743 | 8 | 0 |
| Q100 | Passed | 1.303041 | 7 | 26,849 |
| L2 | Passed | 1.663943 | 4 | 75,310 |
| Full masking M1 | Passed | 2.009176 | 8 | 1,884 |
| Zero control M0 | Passed | 2.110044 | 8 | 0 |

The separate unmodified reference took 1.830002 seconds and predicted eight
segments. Whole-run time was 29.673862 seconds, including all six model loads,
predictions, artifact exports and validation. Environment: Python 3.11.12,
PyTorch 2.7.0+cpu, NumPy 1.26.4, CPU float32, four intra-op and four inter-op
threads. Networking was disabled, and upstream/checkpoint/data mounts were
read-only. Lightning emitted its existing non-fatal `save_hyperparameters`
warning about the network module.

### Prediction and behavior checks

- All arms produced valid 500 x 375 class/segment arrays. Saved NPZ arrays were
  read back and compared with the in-memory predictions. COCO RGB panoptic PNGs
  decoded at the correct size; segment IDs, pixel areas, category IDs, thing
  flags, void encoding and raw/export metadata agreed. Visualization PNGs also
  decoded at the native dimensions. Final logits were finite.
- B0 versus the unmodified `load_model`/`predict_image` path: **exactly equal**
  final mask logits, class logits, semantic pixels and segment pixels. Maximum
  absolute logit difference was 0 for both heads.
- M0 versus B0: **exactly equal** logits and both pixel maps on this image.
  Thus it passed the predefined `rtol=1e-5`, `atol=1e-5` logit tolerance with
  maximum absolute difference 0. All three effective attention masks were
  completely unrestricted, with zero blocked query-to-patch pairs.
- Q100 actually used 100 embedding rows and returned 100 mask/class queries.
  The first nine encoder blocks processed 1,605 tokens and the final three
  processed 1,705, versus B0's 1,805 in its final three blocks.
- L2 retained all 12 encoder blocks. Its first ten blocks processed 1,605 tokens
  and the final two processed 1,805, confirming later query insertion rather
  than deletion of an encoder block.
- M1 used probabilities `[1, 1, 1]`. Blocks 9, 10 and 11 (zero-based) blocked
  310,635, 309,922 and 310,600 of their respective 320,000 query-to-patch pairs.
  No other attention pairs were blocked. M1 and M0 returned four prediction
  stages; only the final stage was exported, as intended.

Read-only hooks observed token counts, attention masks and final logits while
delegating to the original upstream attention and existing prediction function.
They were removed after each prediction. Prediction times include this small
observation/copy overhead, are single cold calls, and exclude model loading and
export. **They are smoke timings, not stable speedup benchmarks.** Changes in
segment counts or semantic pixels establish different behavior, not better or
worse accuracy. This one-image result does not establish dataset-wide equivalence
or generalize performance to 500 or 5,000 images.

### Saved artifacts and preservation

Results are under `results/ablations/smoke-test`:

- `summary.json`: parent subset IDs,
  selected image, hashes, runtime, timing, settings, observed behavior and numeric
  comparisons for every arm and the unmodified reference.
- Each of `baseline`, `queries100`, `blocks2`, `mask-full`, `mask-zero`, and
  `unmodified-reference` contains `prediction.npz`, raw `prediction.json`,
  `panoptic.png`/`panoptic.json`, visual `segmentation.png`/`overlay.png`, and
  `verification.json`.
- `preservation.json`: before/
  after hashes for the checkpoint and existing full-run metric/verification
  records, plus upstream cleanliness.

The checkpoint checksum remained
`7365b41a8a129afc74c12e5dd7af534146286d02822e4b89b7bbe8ea613ff74d`.
The upstream checkout remained clean. The smoke output directory was new and
existing results were not overwritten. The smoke runner refuses an existing
destination; a future rerun needs a fresh `--output-dir`.

**No intervention failed and no framework fix was required.** The only code
addition in this phase was the reproducible single-image smoke runner. This is
validation of fixed-checkpoint inference interventions, not independent training
of query-count, processing-depth or mask-schedule variants.

## Measured 500-image COCO experiments

All five arms completed on 2026-11-09 in the existing CPU Docker image. Outputs
are under `results/ablations/coco-val500`.
Each arm used exactly the same 500 unique IDs selected by
`numpy.default_rng(0).choice(sorted(validation_ids), 500, replace=False)`, then
sorted. The first selected ID is 1353. The source JSON contains 5,000 images,
but **each arm evaluated only 500**.

### Reproducible suite command

The executed command was:

```powershell
docker compose run --rm --no-deps --pull never cpu python -u /workspace/scripts/run_ablation_suite.py
```

The [suite runner](../scripts/run_ablation_suite.py) creates a new parent
directory, executes B0, Q100, L2, M1 and M0 sequentially, and invokes the
[independent verifier](../scripts/verify_ablation.py) in a separate process after
each arm. It stops on failure and preserves all completed/partial outputs. No
run failed in this suite. The runner also generates the comparison CSV/JSON and
checks protected hashes after completion. It never requests a full-validation
ablation or training. A repeat requires a fresh destination, for example
`--output-dir /results/ablations/coco-val500-repeat1`; do not run it again merely
to recreate the already saved files.

The individual-arm settings are listed above. Independent scoring commands were
equivalent to the following for each fresh arm directory:

```text
python /workspace/scripts/verify_ablation.py /results/ablations/coco-val500/baseline
python /workspace/scripts/verify_ablation.py /results/ablations/coco-val500/queries100
python /workspace/scripts/verify_ablation.py /results/ablations/coco-val500/blocks2
python /workspace/scripts/verify_ablation.py /results/ablations/coco-val500/mask-full
python /workspace/scripts/verify_ablation.py /results/ablations/coco-val500/mask-zero
```

Verification outputs use exclusive creation and will not overwrite existing
verification reports. Exact child-process arguments, timestamps, exit handling
and elapsed process timers are preserved in
`suite.json`, with per-arm evaluation
and verification logs beside it.

### Metric results

Values are metric points (raw arm JSON uses 0–1). Differences use the matched
500-image B0 baseline, not the earlier full-set PQ or a paper/model-zoo number.

| Arm | PQ | SQ | RQ | Things PQ | Stuff PQ | Signed PQ difference | Absolute PQ difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline B0 | 47.7820 | 81.6096 | 57.1361 | 50.8259 | 43.3024 | 0.0000 | 0.0000 |
| Q100 | 40.4920 | 74.1415 | 48.1849 | 48.5092 | 28.6931 | -7.2900 | 7.2900 |
| L2 | 28.8466 | 68.1019 | 34.3289 | 29.9015 | 27.2940 | -18.9355 | 18.9355 |
| Full masking M1 | 46.4318 | 80.3132 | 55.3781 | 50.1985 | 40.8885 | -1.3502 | 1.3502 |
| Zero control M0 | 47.7820 | 81.6096 | 57.1361 | 50.8259 | 43.3024 | 0.0000 | 0.0000 |

Full-precision aggregate scores are published in
[verified_metrics.csv](../publication/results/verified_metrics.csv). The original
`comparison.csv` and `comparison.json`, including signed and absolute PQ-point
differences, group metrics and runtime summaries, remain in the local run archive.
All arms had the same 131 active categories: 78 things and all 53 stuff classes.
Toaster (COCO ID 80) and hair drier (ID 89) had no TP/FP/FN and were excluded
under the official scorer's active-category rule. These are subset results,
not full COCO validation scores.

### Measured timing and clock limitation

Each arm used a fresh Python process, CPU float32, four intra-op and four
inter-op threads. Runtime records show Python 3.11.12, PyTorch 2.7.0+cpu and
NumPy 1.26.4. Model/configuration, 640 x 640 resize/pad, thresholds 0.8/0.8,
category mapping and official COCO scoring were held fixed.

| Arm | Inference sum (s) | Mean/image (s) | Median/image (s) | p95/image (s) | Evaluator timer (s) | UTC elapsed (s) |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 822.246168 | 1.644492 | 1.589670 | 1.972252 | 898.393748 | 898.320789 |
| Q100 | 661.659610 | 1.323319 | 1.320269 | 1.417721 | 726.471654 | 726.391412 |
| L2 | 777.632055 | 1.555264 | 1.538615 | 1.728076 | 847.979255 | 866.782523 |
| M1 | 1054.911042 | 2.109822 | 2.090391 | 2.354513 | 1125.930518 | 1125.785708 |
| M0 | 978.478540 | 1.956957 | 1.924682 | 2.179926 | 1046.849078 | 4145.089252 |

The evaluator uses `time.perf_counter()` for per-image and total durations; UTC
elapsed is calculated independently from saved start/finish timestamps. The
timer total covers model setup, prediction, validation/export and scoring, but
excludes the subsequent independent verifier. Per-image durations cover the
existing prediction function, without the smoke test's observation hooks.
Every arm retains its complete 500-row `timings.csv` and matching embedded
per-image records.

**There was a timing anomaly during M0:** one polling call took approximately
3,053 wall-clock seconds, and M0's UTC interval exceeds its evaluator timer by
3,098.240174 seconds (about 51.64 minutes). L2 also shows a smaller 18.803268-second
UTC/timer difference. The exact host/VM suspension or clock-adjustment cause was
not established. No timing was discarded, adjusted or replaced, and no arm was
rerun. Consequently, report both clocks and treat these timings as observational,
not a controlled speed benchmark. In particular, M0 took **17.45 minutes on the
evaluator timer and 69.08 minutes by its UTC timestamps**.

The suite timer recorded 4,811.876295 seconds (80.20 minutes), while its UTC
start/finish interval was 7,929.076610 seconds (132.15 minutes). These scopes
include sequential evaluation and verification, excluding Docker startup.
`cross_arm_verification.json`
records the clock audit, physical PNG counts and cross-arm ID checks separately
without modifying the original run records.

### Coverage, artifact integrity and independent rescoring

For **every arm**, the verifier confirmed 500 unique expected IDs, matched
image/annotation/prediction coverage, and valid original-size RGB prediction
masks. A separate cross-arm audit counted exactly 500 physical PNG files per
arm. Segment IDs, areas, official categories, input/GT/checkpoint/config hashes,
metric CSV/JSON and per-image timings passed checks.

Saved predictions were independently rescored by importing the unmodified
vendored official `panopticapi` worker and category averaging in a **separate
process**, without importing EoMT, the inference wrapper or evaluation wrapper.
The scorer is pinned to `7bb4655548f98f3fedc07bf37e9040a992b054b0`.
All All/Things/Stuff PQ/SQ/RQ values agreed within absolute tolerance **1e-12**;
per-class TP/FP/FN agreed exactly and IoU sums agreed within 1e-8. This is
independent execution on saved masks using the same reference implementation,
not a second independent implementation of PQ.

Each arm contains `verification.json`, `rescored_metrics.json`, and a
`prediction_manifest.json` of mask SHA-256 hashes. Separate rescoring durations
were 10.926908 s (B0), 14.735811 s (Q100), 10.598516 s (L2), 10.914618 s (M1), and
12.700725 s (M0); complete verification also includes decoding and provenance
checks, so its total durations are longer.

**M0 and B0 are byte-identical on all 500 prediction PNGs and on the complete
`predictions.json`**, including all segment category/area assignments. Both JSON
files have SHA-256
`bd199df383f1b82db8998378160e916ed93d7fef8673dd7336be58afe26fcbfd`.
This establishes saved-prediction equivalence throughout this subset. Final
logits were not retained/compared for all 500 images; exact-logit equality was
established only in the earlier single-image smoke test.

### Findings and limits

- Removing the selected 100 learned queries reduced PQ by 7.2900 points. The
  observed Stuff PQ loss was larger than the Things PQ loss. This measures
  reliance on this particular seed-0 query subset; it does not establish the
  achievable accuracy of an architecture trained with 100 queries.
- Introducing queries one block later reduced PQ by 18.9355 points. This
  demonstrates sensitivity to changing the processing path without adapting
  weights. It does not reproduce the paper's independently trained block-count
  ablation or show that training an L2=2 architecture must have this result.
- Reactivating full masking reduced PQ by 1.3502 points on the already annealed
  checkpoint. The zero control preserved predictions while computing the extra
  heads/masks. These observations concern inference interventions; they do not
  isolate the benefit of attention-mask annealing during training.
- This is one uniformly sampled subset, one retained-query selection and one
  ordered execution per arm. No training-seed replicates, paired confidence
  intervals, unseen test split or runtime repetitions were produced. Differences
  are measured point estimates, without a statistical-significance claim.

All prior full-evaluation and smoke-test results were preserved. The suite's
before/after hashes for the checkpoint, full-run metrics/verification and smoke
summary match. The upstream working tree remained clean at
`aa017c34afbe023de9462b17ae6067b52fef9867`; the checkpoint SHA-256 remains
`7365b41a8a129afc74c12e5dd7af534146286d02822e4b89b7bbe8ea613ff74d`.
No full 5,000-image ablation, training, dataset download or model download was
performed.

## Fresh installations

The suite no longer requires the historical full-baseline evaluation, verification or smoke-test files. Pass `--check-historical-integrity` to require all three records and compare their SHA-256 hashes before and after the suite, as the historical runner did. Checkpoint identity remains mandatory in both modes. Prerequisites are checked before the output directory is created. Each arm still undergoes independent saved-mask verification. Use a new output destination; original results are not distributed.
