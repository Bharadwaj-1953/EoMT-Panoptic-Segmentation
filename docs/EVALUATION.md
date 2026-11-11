# COCO panoptic evaluation

Prepared 2026-11-08. Implementation and synthetic tests are complete. **The official
COCO 2017 validation dataset is now downloaded, extracted, and verified; see the
acquisition record below. The first 10-image CPU sanity-check evaluation has now
succeeded, followed by a successful full 5,000-image CPU evaluation. Actual
measured results and independent rescoring are recorded below.** Upstream source
remains unchanged.

## Dataset required for evaluation

Use the **COCO 2017 validation split**, with all 5,000 images for full evaluation.
The [official download listing](https://cocodataset.org/#download) provides:

| Archive | Official URL | Published approximate download size |
|---|---|---|
| Validation JPEGs | http://images.cocodataset.org/zips/val2017.zip | 1 GB |
| Panoptic annotations | http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip | 821 MB |

These official URLs were downloaded during the subsequent dataset acquisition
phase recorded below. Archive sizes above are published estimates; measured byte
sizes are in the acquisition record. Allow additional disk space for extraction
and prediction masks.

The panoptic archive includes `annotations/panoptic_val2017.json` and an inner
`annotations/panoptic_val2017.zip`. Extract the inner validation mask archive too.
Use this final layout (the evaluator reads extracted files, not ZIP archives):

```text
data/coco/
  val2017/
    000000000139.jpg
    ...
  annotations/
    panoptic_val2017.json
    panoptic_val2017/
      000000000139.png
      ...
```

Keep filenames from the JSON. Required JSON metadata:

- `images`: numeric `id`, `file_name`, `width`, `height`.
- `annotations`: `image_id`, mask `file_name`, and `segments_info` containing
  `id`, `category_id`, `area`, `iscrowd`.
- `categories`: official 133 categories with `id`, `isthing` (80 things and 53
  stuff categories); retain `name` and other original metadata.
- Retain image license/source fields and the top-level `licenses`/`info` records
  from the official JSON where supplied.

Ground-truth PNGs encode **segment IDs**, not class colors: ID = R + 256 G +
65,536 B. ID 0 is void; category IDs come from `segments_info`. Do not substitute
instance masks, semantic class-index masks, or visualization PNGs. Detection,
caption, keypoint, train-image, and test-image archives are unnecessary. The
separate `annotations_trainval2017.zip` is not needed for this workflow.

For a subset, keep the full validation JSON and provide at least every selected
JPEG/PNG pair. Only selected files must be present. Missing selected files cause
an error; the evaluator never silently skips them or changes the subset.

## Licensing and attribution

COCO annotations are **CC BY 4.0**, attributed to the COCO Consortium. Image
copyright remains with the individual image owners; COCO does not provide a
single blanket image license. Follow the image-specific license/source records
and applicable Flickr terms. See [COCO terms of use](https://cocodataset.org/#termsofuse)
and the [official terms source](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm).

The model and EoMT source credits/licenses are recorded in [INFERENCE.md](INFERENCE.md).
The COCO scorer is BSD 2-clause, copyright 2018 Alexander Kirillov; its original
notice is retained in `scripts/vendor/panopticapi/LICENSE`. Dataset and result
directories remain ignored by Git except their existing empty markers.

## Upstream inspection and scoring choice

Audited EoMT revision: `aa017c34afbe023de9462b17ae6067b52fef9867`.

- `datasets/coco_panoptic.py`: validation JPEGs, nested validation panoptic mask
  ZIP, validation JSON, RGB-to-segment decoding, official category mapping.
- `training/mask_classification_panoptic.py::eval_step`: resize/pad, forward,
  final mask resize, restoration to original resolution, panoptic conversion.
- `training/lightning_module.py`: `init_metrics_panoptic`,
  `update_metrics_panoptic`, and target conversion. Upstream uses TorchMetrics
  PQ state with custom matching/crowd/void updates.

The upstream metric code is **not identical** to the COCO reference scorer: it
adds class 133 as a stuff class and can suppress a false negative when predicted
void covers more than half of a target. This evaluator instead calls the actual
[COCO reference implementation](https://github.com/cocodataset/panopticapi/blob/7bb4655548f98f3fedc07bf37e9040a992b054b0/panopticapi/evaluation.py)
for official COCO PQ semantics. Do not assume scores are numerically identical
to upstream's custom logged metric or the published model-zoo number.

Unmodified `evaluation.py`, `utils.py`, and package marker are vendored under
`scripts/vendor/panopticapi/`, pinned to revision
`7bb4655548f98f3fedc07bf37e9040a992b054b0`. No installation or network access is
needed to score. Only scorer source files and their license were fetched during
preparation; its sample datasets were not fetched.

## Evaluation method

`scripts/evaluate_panoptic.py` uses the existing `load_model` and `predict_image`
functions in `scripts/infer_panoptic.py`. It fixes EoMT-S DINOv2 COCO panoptic
640×640 2x, the original local checkpoint, CPU float32, seed 0, disabled masked
attention, and mask/overlap thresholds 0.8/0.8. No training or augmentation is
performed. Inputs use the original EoMT resize/pad, normalization, original-size
restoration, and panoptic postprocessing methods. One image is processed at a
time, at its original output resolution.

The default subset is the first 10 image IDs in ascending numeric order, independent
of JSON order and filesystem enumeration. `--limit N` changes the size; if N
exceeds available metadata, all listed images are selected and the actual count
is recorded. `--full` selects all IDs and requires exactly 5,000 metadata images.
Selection is deterministic, not random or stratified; a small subset is a workflow
check and cannot represent full-validation quality. Counts alone cannot establish
dataset authenticity: use the official archives and preserve their metadata.

Predictions map EoMT's contiguous class IDs back to official COCO category IDs.
Segment IDs are remapped to positive RGB-encodable IDs; predicted void becomes
ID 0. Ground-truth crowd flags are preserved. Input dimensions, ground-truth
PNG IDs/areas, predicted labels, and metadata consistency are validated. Ground
truth is scored at native resolution and never interpolated. Unlike upstream's
dataset loader, this workflow rejects mismatched GT dimensions rather than
resizing masks. Dataset files/checkpoints are opened read-only.

The official scorer matches same-category non-crowd segments at **IoU > 0.5**,
excludes ground-truth void from the matching union, ignores crowd targets, and
suppresses unmatched predictions with more than half their area in void or
same-category crowd. Missed non-crowd targets count as false negatives, including
when the prediction is void. Accumulate IoU sums and TP/FP/FN over all selected
images before computing per-class values:

```text
PQ = IoU_sum / (TP + 0.5 FP + 0.5 FN)
SQ = IoU_sum / TP                (0 when TP is 0)
RQ = TP / (TP + 0.5 FP + 0.5 FN)
```

Report macro averages over active categories for All/Things/Stuff; categories
with TP+FP+FN=0 are excluded from averages. Group `n` records the included class
count. An entirely empty group is JSON `null` / blank CSV, with n=0. This wrapper
handles that case explicitly because the reference average divides by zero.
All exported metrics use **0–1**, so multiply by 100 for conventional PQ points.
This is dataset-level accumulation, not an average of per-image PQ scores.

## Reproducible evaluation commands

Run from `path/to/EoMT-Panoptic-Segmentation` after the dataset is obtained and extracted.
The existing CPU image is sufficient; Compose mounts scripts and tests, and
networking is disabled. No rebuild is required for this implementation.

Default small deterministic subset:

```text
docker compose run --rm --no-deps --pull never cpu python /workspace/scripts/evaluate_panoptic.py --images data/coco/val2017 --panoptic data/coco/annotations/panoptic_val2017 --annotations data/coco/annotations/panoptic_val2017.json --checkpoint checkpoints/coco_panoptic_eomt_small_640_2x/pytorch_model.bin --device cpu --limit 10 --output-dir results/evaluation/coco-val2017-first10
```

For a different subset size, change `--limit` and choose a fresh output directory.
Generic full-validation example (the actual completed run used `coco-val5000`,
with its exact command recorded in the final section):

```text
docker compose run --rm --no-deps --pull never cpu python /workspace/scripts/evaluate_panoptic.py --full --device cpu --output-dir results/evaluation/coco-val2017-full
```

All dataset and checkpoint paths can be overridden; relative paths resolve from
the project root and translate to Compose's `/data`, `/checkpoints`, `/results`
mounts. Absolute container paths also work. Only CPU is currently supported.
Existing output directories are rejected. On error, the CLI exits nonzero and
does not write a successful `metrics.json`; partial prediction files may remain.
Fix the cause and use a new output directory. There is no automatic downloading,
resuming, skipping, or retries.

Safe CLI inspection (executed during preparation):

```text
docker compose run --rm --no-deps --pull never cpu python /workspace/scripts/evaluate_panoptic.py --help
```

## Export schema and measurements

Successful runs create these files below the requested `results/evaluation/` run:

| Output | Contents |
|---|---|
| `metrics.json` | Complete status, dataset and evaluated sizes, ordered IDs, model/checkpoint/config identity and hashes, scorer revision, runtime, device, thresholds, load/inference/total durations, aggregate/per-class metrics, per-image records |
| `metrics.csv` | All/Things/Stuff PQ, SQ, RQ, included category count |
| `per_class.csv` | COCO category/name/thing flag, PQ/SQ/RQ, IoU sum, TP/FP/FN, inclusion flag |
| `timings.csv` | Image ID/name/resolution, inference seconds, input/GT SHA-256 |
| `predictions.json` | COCO prediction annotations with image IDs, PNG names, segment/category IDs and areas |
| `panoptic_predictions/*.png` | RGB-encoded segment IDs; not colored visualization masks |
| `ground_truth_subset.json` | Original metadata restricted to selected images/annotations, preserving categories and available credits/licenses |

Per-image `perf_counter` timing includes preprocessing, forward, postprocessing,
and NumPy conversion; excludes image decoding, scoring, export, and hashing.
Model-load timing includes imports/construction/checkpoint loading. Total duration
starts at evaluator entry and includes metadata checks, model loading, image I/O,
inference, scoring, hashing, and auxiliary exports. It ends immediately before
writing `metrics.json`, excluding Docker startup and final JSON serialization.
No warm-up is run; first-image timing may include initialization overhead. These
measurements are not a controlled performance benchmark. The upstream revision
field is the wrapper's expected commit, not a runtime Git audit; verify separately.

## Automated verification

```text
docker compose run --rm --no-deps --pull never --workdir /workspace cpu python -m unittest tests.test_evaluation tests.test_inference -v
```

Final result on 2026-11-08: **22 tests passed, zero skipped**, 7.657 seconds:
12 new evaluation tests and 10 existing inference tests. Synthetic cases cover
perfect PQ; nonperfect SQ/RQ; wrong classes; missing segments; strict 0.5 IoU;
void union correction and FP suppression; crowd matching and same-category
suppression; category macro averaging; accumulation; empty groups; RGB export;
deterministic selection; missing files; duplicate IDs; full-count guard;
ground-truth area checks; invalid encodings; path confinement; and overwrite
rejection. The end-to-end test uses a fake loader/predictor on synthetic 2×2
images and verifies JSON/CSV/PNG exports, then removes its fixtures. It never
loads the pretrained checkpoint. Existing integration tests exercise official
EoMT preprocessing/postprocessing helpers with synthetic tensors.

The initial run exposed NumPy scalar mutation by the official scorer, causing
prediction JSON serialization to fail. The wrapper now scores a deep copy of
prediction metadata; the end-to-end test confirms exports succeed. No vendored
scorer changes were needed. Existing Matplotlib deprecation warnings were non-fatal.

Real subset/full PQ and full-run resource usage remain **unverified** until the
evaluation phase. Dataset integrity was subsequently verified as recorded below. No training,
fine-tuning, or real quantitative dataset evaluation was run during preparation.

## Official validation dataset acquisition and verification

Completed 2026-11-08 America/New_York (UTC verification timestamp:
`2026-11-09T03:33:48.604706+00:00`). Both official HTTP downloads returned status 200
and curl exit code 0. No download retries or alternative sources were needed.

| Archive / retained file | Exact bytes | Source |
|---|---|---|
| `data/downloads/val2017.zip` | 815,585,330 | http://images.cocodataset.org/zips/val2017.zip |
| `data/downloads/panoptic_annotations_trainval2017.zip` | 860,725,834 | http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip |
| `data/downloads/panoptic_val2017.zip` | 33,977,091 | Extracted unchanged from `annotations/panoptic_val2017.zip` in the panoptic archive |

Download durations reported by curl: 25.508456 seconds for validation images,
26.016638 seconds for panoptic annotations. These are one-run observations.
The annotation archive officially bundles train and validation annotations;
training annotations remain compressed in that retained archive and were not
extracted. **No COCO training images were downloaded.**

Locally computed SHA-256 values (provenance fingerprints, not a comparison to
independently published publisher checksums):

| File | SHA-256 |
|---|---|
| `val2017.zip` | `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05` |
| `panoptic_annotations_trainval2017.zip` | `c05f76d2129b6b561eb70efe16e7006df62f73fb92889132d373b9d90e31a370` |
| Inner `panoptic_val2017.zip` | `1c496c622b6a6620bacf7e2085d5fe7882e0cf3e8835fd68844276b4b13edde5` |
| `panoptic_val2017.json` | `454873a8a01114246066ac841750eb742df3b5e42ce927ef38b49690084ec75a` |

Only `val2017/*.jpg`, `annotations/panoptic_val2017.json`, and
`panoptic_val2017/*.png` were extracted into the layout shown above. The nested
mask ZIP was retained in `data/downloads/`. macOS resource-fork entries and
`.DS_Store` metadata were excluded. Extraction used Python's `zipfile` member
streams, which checked CRCs on all extracted members. Existing files were not
overwritten; resolved extraction destinations were confined to the intended
validation directories.

| Verification | Observed result |
|---|---|
| Validation JPEG files | **5,000**, 814,705,164 bytes total |
| JSON `images` entries | **5,000 unique image IDs** |
| JSON `annotations` entries | **5,000 unique `image_id` values** |
| Validation panoptic PNG files | **5,000**, 35,169,243 bytes total |
| Validation JSON size | 7,005,201 bytes |
| Extracted dataset payload | **856,879,608 bytes** (JPEGs + PNGs + JSON; excludes archives/manifest) |
| Categories | **133**: 80 things, 53 stuff |
| Ground-truth segment records | **56,728** |
| Image/annotation ID sets | Exact equality; no missing or duplicate IDs |
| Filenames | JPEG and PNG filename stems match corresponding numeric image IDs; exact file-set agreement with JSON |
| Readability | All 5,000 JPEGs and all 5,000 PNGs fully decoded successfully |
| Dimensions | Every JPEG and RGB mask matches JSON width/height |
| Ground-truth integrity | Every nonzero PNG segment ID and pixel area matches `segments_info`; all category IDs/crowd flags valid |
| Git exclusion | All **10,006** non-marker files currently under `data/` ignored, including the existing coffee sample; zero tracked data files |
| Upstream | Clean at `aa017c34afbe023de9462b17ae6067b52fef9867` |
| Evaluation / training | **Not run** |

The panoptic annotation record's image link is `image_id`; segment `id` fields
identify segments within each mask and are not expected to equal the image ID.
The ID checks above concern image IDs and annotation `image_id` values.

Dataset integrity verification took **138.708870 seconds**, excluding downloads
and extraction. A machine-readable acquisition/verification manifest is retained
at `data/coco/DATASET_VERIFICATION.json` (ignored by Git). Git exclusions were
checked for every data file with `git check-ignore --stdin -z`; `git ls-files --
data` returned no tracked files. Existing `.gitignore` rules were sufficient and
were not changed.

The default deterministic first-10 subset now resolves to these image IDs:
`139, 285, 632, 724, 776, 785, 802, 872, 885, 1000`.
This list was read from the verified metadata; inference/PQ scoring was not run.
No evaluator, model inference, training, or fine-tuning was started during dataset
acquisition. Stop after successful dataset verification; the evaluation commands
above remain unexecuted on COCO.

## First quantitative run: 10-image sanity check

**This is a deterministic 10-image sanity check, not a paper reproduction result
or a full-validation quality estimate.** The official EoMT-S checkpoint loaded
and the first experiment completed with exit code 0. No evaluator modifications,
retries, or inference reruns were required.

Exact command executed from the project root:

```text
docker compose run --rm --no-deps --pull never cpu python /workspace/scripts/evaluate_panoptic.py --images data/coco/val2017 --panoptic data/coco/annotations/panoptic_val2017 --annotations data/coco/annotations/panoptic_val2017.json --checkpoint checkpoints/coco_panoptic_eomt_small_640_2x/pytorch_model.bin --device cpu --limit 10 --output-dir results/evaluation/coco-val10
```

The output directory must not already exist. To repeat, use the same command
with a fresh output directory. No rebuild, package install, or download was
needed. Compose disabled networking; the checkpoint, input data, and upstream
source were mounted read-only. The 5,000-image option was not used.

Selection: exactly the first **10** numeric IDs from the verified 5,000-image
validation metadata, in order: `139, 285, 632, 724, 776, 785, 802, 872, 885, 1000`.
The completion file timestamp (UTC) is `2026-11-09T03:37:13.160854+00:00`.

### Actual measured quality

Numbers below are **0-100 points**, obtained by multiplying the raw 0-1 exports
by 100. Category counts indicate active categories included in each macro mean.

| Group | PQ | SQ | RQ | Included categories |
|---|---|---|---|---|
| All | 38.458081 | 47.757844 | 44.906136 | 52 |
| Things | 42.330067 | 51.342640 | 49.585714 | 25 |
| Stuff | 34.872908 | 44.438589 | 40.573192 | 27 |

Scoring used the integrated official COCO panoptic reference implementation at
revision `7bb4655548f98f3fedc07bf37e9040a992b054b0`. Counts accumulated over the
10 images were **63 TP, 25 FP, and 53 FN** across included classes, with 100
exported prediction segments. The remaining unmatched predictions were ignored
by the scorer's crowd/void rules. These counts are not instance detection AP.

### Actual measured timing

| Measurement | Seconds |
|---|---|
| Total evaluation duration | **31.233599** |
| Sum of per-image inference durations | 17.454644 |
| Average inference duration per image | **1.745464** |
| Model load duration | 3.298646 |

Total duration uses the scope defined above: evaluator entry through scoring and
auxiliary exports, immediately before final `metrics.json` serialization; it
excludes Docker startup, post-run verification, and documentation work. Per-image
inference includes preprocessing, forward, and postprocessing, excluding I/O and
scoring. No warm-up or benchmark repetitions were performed.

| Image ID | Input width x height | Inference seconds |
|---|---|---|
| 139 | 640 x 426 | 2.333938 |
| 285 | 586 x 640 | 1.722709 |
| 632 | 640 x 483 | 1.960945 |
| 724 | 375 x 500 | 1.613458 |
| 776 | 428 x 640 | 1.441409 |
| 785 | 640 x 425 | 1.602904 |
| 802 | 424 x 640 | 1.613474 |
| 872 | 621 x 640 | 1.667051 |
| 885 | 640 x 427 | 1.620469 |
| 1000 | 640 x 480 | 1.878287 |

### Hardware and model provenance

- Host CPU: **11th Gen Intel Core i7-11390H @ 3.40 GHz**, 4 physical cores /
  8 logical processors. CPU clock is the reported model designation, not a
  measured sustained operating frequency.
- Docker Desktop Linux x86_64 / WSL2 kernel `6.18.54.1-microsoft-standard-WSL2`;
  8 exposed logical CPUs, 8,193,245,184 bytes Docker engine memory.
- PyTorch intra-op/inter-op threads: **4 / 4**, observed with a separate probe of
  the same image and Compose service. Device **CPU**, precision **float32**.
- Python `3.11.12`, PyTorch `2.7.0+cpu`, NumPy `1.26.4`; Docker Engine `29.9.2`.
- CPU image ID: `sha256:c2c74a4e9447ff14d6905db6aa335bffd5fb1fa2434bd3fe4558092851f41832`.
- Model: official `tue-mps/coco_panoptic_eomt_small_640_2x`, EoMT-S DINOv2,
  640x640 input preprocessing, 200 queries, final-layer panoptic predictions,
  mask/overlap thresholds 0.8/0.8.
- Local checkpoint: `checkpoints/coco_panoptic_eomt_small_640_2x/pytorch_model.bin`.
  Runtime SHA-256 matched the previously verified official original-format file:
  `7365b41a8a129afc74c12e5dd7af534146286d02822e4b89b7bbe8ea613ff74d`.
- Upstream source remained clean at
  `aa017c34afbe023de9462b17ae6067b52fef9867` before/after the run.

### Output verification and warnings

All outputs are under `results/evaluation/coco-val10/`:

- `metrics.json`, `metrics.csv`, `per_class.csv`, `timings.csv`.
- `predictions.json`, `ground_truth_subset.json`.
- `panoptic_predictions/000000000139.png` and the other nine ID-named PNGs.
- `verification.json`: post-run integrity checks and rescoring comparison.
- `hardware.json`: hardware observations, run label, and exact command.
- `evaluation.log`: captured warning, evaluator progress, and printed metrics
  (Docker startup lines excluded).

All 10 PNGs fully decoded as RGB at their corresponding native input dimensions.
Positive PNG IDs matched JSON segments exactly, pixel areas matched metadata,
and all categories were valid. The subset JSON, prediction JSON, timing records,
and requested IDs matched exactly, with no skips. CSV and JSON metric/timing
values agreed; checkpoint/config/annotation and selected image/GT hashes matched
local files. All metrics were finite and in the valid range.

The exported masks were rescored without model inference using the official
scorer. Group PQ/SQ/RQ matched; per-class scores/counts agreed within absolute
numeric tolerance `1e-12`. The initial validation used exact float equality and
flagged a per-class accumulation-order difference of
`3.552713678800501e-15`; tolerance-based verification passed. This was a post-run
validation comparison issue, not an evaluator failure. No stored metric values
were changed.

The sole evaluation warning was Lightning's existing advisory that `network` is
an `nn.Module` already saved during checkpointing and could be excluded from
`save_hyperparameters`. It did not prevent loading or evaluation. No runtime
errors or NaN/Inf values were observed.

SQ is a macro mean over all active classes, including **23 active classes with
no successful match** whose SQ is zero. That explains why the macro SQ is lower
than matched-segment-only IoU summaries. Macro PQ need not equal macro SQ times
macro RQ. Predicted void pixels were present, were not discarded from validation,
and were handled by the official metric. The subset is small and unstratified;
these scores cannot be used to assert agreement with published full-validation
results. Per-class failures are retained in `per_class.csv` for inspection.

No training, fine-tuning, or full 5,000-image evaluation was performed. The run
and output verification are complete; stop here.

## Full COCO val2017 evaluation: 5,000 images

**Completed successfully on 2026-11-09, exit code 0.** This is the measured full
validation result for the official pretrained EoMT-S checkpoint in the documented
CPU float32 environment. The earlier preparation/10-image sections are historical;
this record supersedes their statements that full evaluation had not been run.
No model, evaluator, preprocessing, threshold, or upstream-source changes were
made for this run. No training or fine-tuning occurred. Existing results were
preserved; no results were deleted or overwritten.

### Preflight and exact command

Preflight confirmed exactly 5,000 unique metadata IDs and all 5,000 local
image/annotation file pairs, a new output directory, a readable original-format
checkpoint, the CPU image, and a clean upstream checkout. Available disk space
was **203,869,794,304 bytes** before launch and
**199,580,594,176 bytes** after the Docker command completed.
No image rebuild or download was needed.

Exact command executed from the project root:

```text
docker compose run --rm --no-deps --pull never cpu python /workspace/scripts/evaluate_panoptic.py --images data/coco/val2017 --panoptic data/coco/annotations/panoptic_val2017 --annotations data/coco/annotations/panoptic_val2017.json --checkpoint checkpoints/coco_panoptic_eomt_small_640_2x/pytorch_model.bin --device cpu --full --output-dir results/evaluation/coco-val5000
```

`--full` selected all 5,000 IDs in ascending numeric order. Exactly 5,000 unique
images were evaluated; there were no skips or duplicate image records. `--limit`
was not supplied. Inputs/checkpoint/upstream remained read-only in Compose, with
container networking disabled.

Run timestamps (UTC): `2026-11-09T03:47:02.505912+00:00` to
`2026-11-09T06:02:21.245059+00:00`. A repeat run must use a new output directory to
preserve this result.

### Actual full-validation metrics

Values below are **0-100 points**; raw CSV/JSON values remain on the 0-1 scale.

| Group | PQ | SQ | RQ | Included categories |
|---|---|---|---|---|
| All | **46.736418** | 81.533019 | 56.335509 | 133 |
| Things | **50.153632** | 82.000796 | 60.332856 | 80 |
| Stuff | **41.578360** | 80.826940 | 50.301778 | 53 |

The integrated official COCO panoptic scorer accumulated results across all
5,000 images before category averaging. All 133 categories were active: 80
things and 53 stuff. Totals verified by independent rescoring: **30,677 TP,
9,723 FP, and 25,618 FN**, with **43,348 predicted segments**. Void/crowd handling
and metric definitions are unchanged from the methodology above. No statistical
uncertainty estimate or claim of bitwise equivalence to a paper's hardware/runtime
is made; these are actual measurements of this checkpoint and workflow.

### Timing and environment

| Measurement | Seconds |
|---|---|
| Total evaluator duration | 8112.858669 |
| Docker command wall duration | 8118.738812 |
| Sum of image inference durations | 7410.797736 |
| Mean inference duration per image | 1.482160 |
| Median inference duration per image | 1.450791 |
| 95th percentile inference duration | 1.943489 |
| Minimum inference duration | 1.051415 |
| Maximum inference duration | 4.620741 |
| Model load duration | 2.367984 |
| Independent rescore duration | 65.470094 |
| Complete post-run verification duration | 123.675799 |

Total evaluator duration was **2 hours, 15 minutes, 12.859 seconds**. Its scope
is evaluator entry through auxiliary export, immediately before final metrics
JSON serialization. Docker command wall time additionally includes container
startup/exit and final serialization. Verification, rescoring, and documentation
are excluded from evaluation timings. Per-image timing includes original EoMT
preprocessing, CPU forward, postprocessing, and NumPy conversion; it excludes
file I/O, scoring, and hashes. Every measured image duration is retained in
`timings.csv` and `metrics.json`. No warm-up/repeated benchmark runs were used.

- CPU: **11th Gen Intel Core i7-11390H @ 3.40 GHz**, 4 physical cores /
  8 logical processors. The clock is the processor's reported model designation.
- Docker Desktop Linux x86_64 / WSL2 kernel
  `6.18.54.1-microsoft-standard-WSL2`; Docker Engine `29.9.2`; 8 exposed logical
  CPUs and **8,193,241,088 bytes** Docker engine memory.
- PyTorch intra-op/inter-op threads: **4 / 4**, observed in a pre-run probe of
  the same image/Compose service. Device **CPU**, precision **float32**.
- Python `3.11.12`, PyTorch `2.7.0+cpu`, NumPy `1.26.4` in the evaluator.
- Image ID: `sha256:c2c74a4e9447ff14d6905db6aa335bffd5fb1fa2434bd3fe4558092851f41832`.
- Model: `tue-mps/coco_panoptic_eomt_small_640_2x`, original EoMT-S DINOv2,
  640x640 preprocessing, 200 queries, final-layer predictions, 0.8 confidence and
  0.8 overlap thresholds, seed 0.
- Checkpoint SHA-256:
  `7365b41a8a129afc74c12e5dd7af534146286d02822e4b89b7bbe8ea613ff74d`.
- Source remained clean at `aa017c34afbe023de9462b17ae6067b52fef9867`.
- COCO scorer revision: `7bb4655548f98f3fedc07bf37e9040a992b054b0`.

Full command, timestamps, disk space, exit status, CPU/image observations,
runtime details, and SHA-256 fingerprints of wrapper/scorer/Compose sources are
stored in `run.json` and `environment.json`.

### Completeness and independent rescoring

Post-run verification passed for **all 5,000** prediction PNGs and metadata/timing
records. Checks established:

- Exact equality of unique image, annotation, prediction, and timing ID sets
  against all 5,000 original COCO validation IDs; no missing/extra PNGs.
- Every PNG fully decodes as RGB at the corresponding native input dimensions.
- PNG segment IDs, metadata segment IDs, pixel areas, and category IDs agree.
- Checkpoint, config, annotation JSON, and every selected input/GT file hash
  matches the recorded evaluation provenance.
- All CSV and JSON metrics/timings agree. Scores/timings are finite and valid.
- Upstream remains unchanged; all run artifacts remain ignored by Git.

Independent rescoring ran in a **separate host Python process**, importing the
unmodified vendored COCO worker and `PQStat.pq_average` directly. It did not import
our evaluator/model or run inference. It read the original full validation JSON,
original ground-truth PNGs, and exported prediction JSON/PNGs and scored all
5,000 images together. The historical command was:

```text
python results/evaluation/coco-val5000/verify_and_rescore.py
```

For a fresh verification, use `python -m scripts.verification.verify_and_rescore` with `EOMT_VERIFY_RUN` set to the saved full-baseline run and `EOMT_VERIFY_OUTPUT` set to a separate, new output directory. These paths are shell environment variables. The original saved masks and metadata are required locally and are not distributed.

Aggregate PQ/SQ/RQ and per-class scores agreed within absolute tolerance `1e-12`;
TP/FP/FN counts matched exactly. Maximum per-class accumulated IoU-sum difference
was `2.2737367544323206e-11`, within the explicit `1e-8`
IoU-sum tolerance; this reflects float accumulation order. Raw metrics were not
changed. The independent scores are preserved in `rescored_metrics.json`, with
validation details in `verification.json` and each PNG hash in
`prediction_manifest.json`.

No warnings, errors, tracebacks, or NaN/Inf metric values were detected in the
captured full-run log. The full run completed on its first attempt; no restart or
retry was performed. This full run used the same existing evaluator as the sanity
check. No existing results were deleted.

### Output locations

All completed artifacts are under `results/evaluation/coco-val5000/`:

| Artifact | Purpose |
|---|---|
| `metrics.json`, `metrics.csv` | Measured overall/things/stuff PQ, SQ, RQ and evaluation metadata |
| `per_class.csv` | All 133 per-category scores and counts |
| `timings.csv` | All 5,000 per-image inference durations, resolutions, input/GT hashes |
| `predictions.json`, `panoptic_predictions/*.png` | All 5,000 COCO-format prediction records/masks |
| `ground_truth_subset.json` | Metadata for all 5,000 evaluated images (full set) |
| `evaluation.log`, `run.json`, `environment.json` | Execution log, exit status, timestamps, command, hardware/runtime provenance |
| `rescored_metrics.json`, `verification.json`, `prediction_manifest.json` | Independent metric/completeness validation and mask hashes |
| `verify_and_rescore.py`, `verification.log` | Reproducible saved-output verifier and its log |

Original host-side supervisor log/status are also retained as
`results/evaluation/coco-val5000.log` and `results/evaluation/coco-val5000.run.json`.
All previous inference and 10-image results remain present. Evaluation and
verification are complete; no training, model changes, or further experiments
were started. Stop here.
