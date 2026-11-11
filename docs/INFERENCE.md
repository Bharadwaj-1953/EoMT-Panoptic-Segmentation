# Phase 3: offline CPU panoptic inference

Quantitative evaluation preparation is documented in [EVALUATION.md](EVALUATION.md),
including required COCO files, official reference PQ scoring, deterministic subset
selection, execution commands, exports, and synthetic test results. No COCO
dataset has been downloaded or evaluated in that preparation phase.

## Status

Implemented outside `upstream/eomt`; no upstream edits or new dependencies. The wrapper supports local images with the smallest officially listed DINOv2 COCO panoptic EoMT model.

**Windows x86_64 CPU inference tests: PASSED — all 10 tests passed, zero skipped**, including PyTorch integration tests, based on the user's successful Docker build and test run. Matplotlib emitted non-fatal deprecation warnings. The earlier container attempt was blocked by Docker permissions; host-only testing reported 7 passed and 3 skipped. **First real pretrained CPU inference succeeded on 2026-11-08; see the final run record below. The official checkpoint and one CC0 sample image have been downloaded. No dataset evaluation has been run.** macOS ARM64 and NVIDIA GPU testing remain **NOT VERIFIED**.

## Selected checkpoint and attribution

| Item | Selection |
|---|---|
| Official model | EoMT-S, COCO panoptic, 640×640, 2x schedule |
| Hub repository | [tue-mps/coco_panoptic_eomt_small_640_2x](https://huggingface.co/tue-mps/coco_panoptic_eomt_small_640_2x) |
| Original checkpoint | [pytorch_model.bin](https://huggingface.co/tue-mps/coco_panoptic_eomt_small_640_2x/resolve/main/pytorch_model.bin) |
| Expected local path | `checkpoints/coco_panoptic_eomt_small_640_2x/pytorch_model.bin` |
| Official configuration | `upstream/eomt/configs/dinov2/coco/panoptic/eomt_small_640_2x.yaml` |
| Backbone | `vit_small_patch14_reg4_dinov2`, actual segmentation patch size 16 |
| Query processing | 200 queries, last 3 encoder blocks, 133 classes plus no-object |
| Published zoo result | 46.7 PQ, a reported result only; not reproduced here |
| Source revision | `aa017c34afbe023de9462b17ae6067b52fef9867` |

Small is the smallest backbone among the officially listed DINOv2 COCO panoptic S/B/L/g models. The Hub metadata inspected during this phase lists the original binary at approximately 96 MB. During the initial implementation phase, only the model card and file listing were read. The subsequent checkpoint acquisition is recorded below. This is not a measured RAM or CPU throughput requirement.

The config logger name lacks the `_2x` suffix, whereas the official model-zoo checkpoint repository includes it. The wrapper uses the explicit identifier above and never derives a download name from the logger.

Use the original `pytorch_model.bin` linked by the [official DINOv2 model zoo](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/model_zoo/dinov2.md), not the Transformers-converted `model.safetensors` from the same Hub repository. The latter has a different state-dictionary layout. Backbone-only DINOv2 files are also insufficient. The downloaded binary has passed CPU deserialization and tensor integrity checks. Full architecture/key compatibility was subsequently confirmed by the successful pretrained run recorded below; incompatible keys cause an error rather than partially initialized inference.

Licenses and credits:

- Official EoMT source: MIT, copyright 2025 Mobile Perception Systems Lab at TU/e. Preserve [upstream LICENSE](../licenses/EoMT-MIT.txt).
- The selected [Hub model card](https://huggingface.co/tue-mps/coco_panoptic_eomt_small_640_2x) labels the checkpoint MIT. No weights are redistributed here.
- DINOv2 code/pretrained models are attributed to Meta AI Research/FAIR; see [DINOv2 README](https://github.com/facebookresearch/dinov2) and [Apache-2.0 license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).
- EoMT's code retains notices for adapted timm, Mask2Former, Transformers, Lightning, and TorchMetrics code. This project imports its implementations directly; those notices remain unchanged.
- Cite Tommie Kerssies, Niccolò Cavagnero, Alexander Hermans, Narges Norouzi, Giuseppe Averta, Bastian Leibe, Gijs Dubbelman, and Daan de Geus, *Your ViT is Secretly an Image Segmentation Model*, CVPR 2025. The official README contains the BibTeX citation.
- User-provided images retain their own licenses. No COCO images, annotations, or other datasets are bundled.

## Implementation and behavior

Files added: `scripts/infer_panoptic.py`, package marker, `tests/test_inference.py`, test package marker, and this document. Compose mounts scripts/tests read-only and supplies runtime mount mappings. Both Dockerfiles copy the new code for standalone images; the build context allows these paths. Existing image dependencies and default environment smoke commands are unchanged.

The reusable functions are `load_model(checkpoint)`, `predict_image(model, PIL_image)`, and `save_prediction(...)`. The CLI fixes the official small 2x model configuration and CPU device; it is not an arbitrary configuration loader.

The official notebook and loading code were inspected locally:

- [Notebook](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/inference.ipynb): dynamically builds the model, retrieves weights from the Hub, then runs CUDA autocast panoptic inference.
- [ViT construction](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/models/vit.py): passing a non-None local checkpoint path makes timm use `pretrained=False`. This avoids the notebook's extra backbone download.
- [Checkpoint loader](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/training/lightning_module.py): CPU map location, `weights_only=True`, optional state_dict wrapper, known criterion buffer exclusion, and explicit missing/unexpected key checks.
- [Panoptic task wrapper](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/training/mask_classification_panoptic.py): constructs the unchanged model heads and task defaults. No Trainer, dataset module, W&B logger, fit, validation loop, or metric update is invoked.

The wrapper preserves the original architecture and calls official preprocessing/postprocessing helpers:

1. Decode each local image with Pillow and convert to RGB uint8, matching upstream RGB decoding. No extra EXIF orientation transformation is applied.
2. Preserve aspect ratio, resize to fit 640×640 with upstream PIL bilinear resizing, and pad right/bottom using the official helper.
3. Run the unmodified Lightning model forward: divide by 255, then the EoMT network applies ImageNet mean/std. No duplicated normalization.
4. Disable masked attention as in official pretrained evaluation/inference; use final-layer predictions.
5. Bilinearly interpolate masks to the configured size, remove padding, and restore original image dimensions with upstream helpers.
6. Apply upstream panoptic confidence threshold 0.8, overlap threshold 0.8, sigmoid masks, per-pixel query assignment, and same-class stuff merging.

CPU uses float32 under `torch.inference_mode()`, without CUDA autocast or compilation. This preserves the operation sequence, but it does **not** promise bitwise agreement with the notebook's CUDA float16 path. Border rendering follows the notebook's class/segment boundary logic; the deterministic HSV palette and optional overlay are presentation outputs, not model postprocessing changes.

Offline flags are forced before ML imports; no Hub download function is called. Compose also disables container networking. The fixed DINOv2 config and local checkpoint argument prevent automatic pretrained-backbone loading. Missing/empty checkpoints fail before model imports/allocation. A malformed or incompatible checkpoint fails with an error identifying the required source format.

## Commands: weight-free checks now

Run at the project root. These are single-line commands valid in Windows CMD, PowerShell, and macOS Terminal. They use the existing CPU image and read-only mounts, so no rebuild or dependency install is necessary to try the tests:

```text
docker compose run --rm --no-deps cpu python /workspace/scripts/infer_panoptic.py --help
docker compose run --rm --no-deps --workdir /workspace cpu python -m unittest tests.test_inference -v
```

These tests do not require weights. A fully configured CPU container should run all 10 tests without skips. Three integration checks require the existing PyTorch environment:

- Synthetic RGB input and synthetic mask/class logits pass through the actual upstream resize/pad, restore-size, and panoptic helpers, verifying stuff merging and output writing.
- Constructor doubles verify small-model configuration, local loading, and suppressed backbone pretraining. No real backbone is built.
- A mocked incompatible state dict verifies the official loader rejects missing model parameters instead of continuing with random weights.

The remaining tests cover missing-checkpoint messages/exit status, CLI help, relative mount mapping, image selection, output confinement, raw-ID round trips, void pixels, segment/category metadata, visual dimensions/borders, and overwrite rejection.

The test fixtures are generated locally under `results/` and removed afterward. They are synthetic arrays and temporary placeholder files, not downloaded datasets or pretrained weights.

Optional rebuild to include scripts/tests in standalone image layers:

```text
docker compose build cpu
```

This downloads image/package layers as needed, not checkpoints or datasets. It was not needed or completed during this phase.

## Future use with a locally provided checkpoint

The following inference commands are documentation only and have not been executed with the pretrained checkpoint.

The verified original binary is now at the expected local checkpoint path. Inference was subsequently completed, as recorded in the final run section below. For a fresh run, place existing input images, for example `photo.jpg`, in `data/images/`.

```text
docker compose run --rm --no-deps cpu python /workspace/scripts/infer_panoptic.py --input data/images/photo.jpg --output-dir results/inference/run-001
```

Multiple files or a directory are accepted:

```text
docker compose run --rm --no-deps cpu python /workspace/scripts/infer_panoptic.py --input data/images --checkpoint checkpoints/coco_panoptic_eomt_small_640_2x/pytorch_model.bin --output-dir results/inference/run-002
```

Supported extensions: JPEG, PNG, BMP, WebP, TIFF. Directory scanning is nonrecursive and sorted; multiple explicitly supplied paths retain argument order. Images are processed one at a time. Corrupt images produce errors; earlier images in a failed multi-image run may already have outputs.

Relative paths resolve from the project root, independently of the shell's current directory inside the container. Compose maps relative `data/...`, `checkpoints/...`, and `results/...` to `/data`, `/checkpoints`, and `/results`. Absolute container paths also work. All requested CLI outputs must resolve underneath the project's results mount.

Output run directories must be new. Existing directories are rejected to avoid accidental overwrite; choose another run name if retrying. Input files and checkpoints are opened read-only. CPU inference time and RAM use have not been measured.

## Output schema

Each image creates `results/inference/<run>/<index>_<image-stem>/`:

| File | Contents |
|---|---|
| `prediction.npz` | Original-resolution int32 arrays `semantic[H,W]` and `segment_ids[H,W]`; load with NumPy `allow_pickle=False` |
| `prediction.json` | Input path, dimensions, segment areas/classes/category IDs, model/config identity, thresholds, CPU precision, and checkpoint/config SHA-256 hashes |
| `segmentation.png` | RGB class-color visualization with black class/instance boundaries |
| `overlay.png` | 50% blend of RGB input and segmentation visualization |

Class IDs are EoMT's contiguous COCO IDs 0–132: 0–79 are things, 80–132 are stuff. The void class is **133** and the void segment ID is **-1**. Segment IDs are zero-based and may contain gaps. JSON maps each valid segment to its original COCO category ID using upstream `CLASS_MAPPING`; void pixels are excluded from segment metadata. No class-name list or confidence scores are invented. These are inspection outputs, **not** a COCO evaluation submission format.

`source_commit_expected` records the audited revision, not a fresh runtime Git verification; image layers deliberately exclude .git. The config and checkpoint bytes receive hashes when actual inference runs. Verify the source revision separately before interpreting a result. Hub main is mutable; the downloaded checkpoint revision and verified checksum are pinned in the acquisition record below.

## Verification results

Date: 2026-11-08.

### Latest result: user-verified Windows Docker CPU run

Evidence source: the user's report of these commands completing successfully. No raw build/test log or independent container test rerun was available for this record.

```text
docker compose build cpu
docker compose run --rm --no-deps --workdir /workspace cpu python -m unittest tests.test_inference -v
```

| Check | Result |
|---|---|
| CPU Docker image build | **PASSED** |
| Windows x86_64 CPU inference test suite | **PASSED: 10 tests passed, zero skipped** |
| PyTorch integration tests | **PASSED** |
| Matplotlib warnings | Non-fatal deprecation warnings; tests passed |
| macOS ARM64 / NVIDIA GPU tests | **NOT VERIFIED** |
| Real checkpoint loading / pretrained inference | **NOT RUN** |
| Training / quantitative evaluation | **NOT RUN** |

The successful suite verifies the weight-free synthetic/helper, constructor-double, loading-error, and output tests described above. It does not verify real checkpoint compatibility or pretrained segmentation quality. These weight-free tests did not download checkpoints or datasets or start pretrained inference or training.

### Earlier host checks and Docker access failure

| Check actually executed | Result |
|---|---|
| Notebook/source/config/model-zoo inspection | Completed |
| Public checkpoint model card/file-list metadata inspection | Completed; no weight file downloaded |
| `python -m unittest tests.test_inference -v` on host Python 3.13 | **7 passed, 3 skipped** (10 discovered) |
| Host dependency availability | NumPy/Pillow available; torch/timm/Lightning unavailable; no packages installed |
| Direct host CLI invocation with absent default checkpoint | Printed the expected checkpoint path, official source, and explicit no-download message; stopped before loading |
| `docker compose config --quiet` | Passed |
| `docker compose --profile gpu config --quiet` | Passed |
| Container inference smoke test command above | Blocked before execution, exit 1 |
| Real checkpoint loading/pretrained forward | **NOT RUN** |
| Training / quantitative evaluation | **NOT RUN** |
| Upstream working tree | Clean at the verified commit |

Exact container blocker:

```text
permission denied while trying to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

The initial host test attempt also encountered Windows permission errors from Python 3.13's owner-only TemporaryDirectory directories. Test fixtures now use unique directories under the writable results root with inherited permissions; the final host suite passes as listed above. Some empty inaccessible directories from the initial failed attempt may remain under results; they contain no prediction data or weights.

The subsequent reported Docker run passed all weight-free integration tests. Those tests alone did not verify pretrained inference. Checkpoint acquisition and successful pretrained inference are recorded in the following sections; quantitative evaluation is documented separately in [EVALUATION.md](EVALUATION.md).


## Official checkpoint acquisition and verification

Date: 2026-11-08. Download and read-only verification completed successfully.

| Item | Verified value |
|---|---|
| Official repository | https://huggingface.co/tue-mps/coco_panoptic_eomt_small_640_2x |
| Checkpoint filename | `pytorch_model.bin` (original upstream state dictionary) |
| Local path | `checkpoints/coco_panoptic_eomt_small_640_2x/pytorch_model.bin` |
| Pinned Hub revision | `10f5326794a8ce7e0e975791033ae1c13aec3767` |
| Download source URL | https://huggingface.co/tue-mps/coco_panoptic_eomt_small_640_2x/resolve/10f5326794a8ce7e0e975791033ae1c13aec3767/pytorch_model.bin |
| File size | **95,968,930 bytes** |
| SHA-256 | `7365b41a8a129afc74c12e5dd7af534146286d02822e4b89b7bbe8ea613ff74d` |
| Completeness | Exact byte size and SHA-256 match Hugging Face API file/LFS metadata |
| Readability | Existing CPU Docker image: `torch.load(..., map_location="cpu", weights_only=True)` succeeded |
| Tensor integrity | 198 tensor entries, 23,976,085 elements; all tensors finite |
| Git exclusion | `git check-ignore -v` confirms the checkpoint is excluded by `.gitignore` rule `checkpoints/*` |
| Upstream source | Clean at `aa017c34afbe023de9462b17ae6067b52fef9867`; no upstream files modified |

The original checkpoint was selected from the link in `upstream/eomt/model_zoo/dinov2.md`. The same Hub repository also contains Transformers-converted `model.safetensors`, which was not downloaded. Metadata was verified using https://huggingface.co/api/models/tue-mps/coco_panoptic_eomt_small_640_2x?blobs=true .

Verification only deserialized the checkpoint on CPU and inspected its tensors; it did not construct the model or execute a forward pass. Full model loading compatibility, pretrained inference, training, and evaluation remain untested. No packages, models other than this checkpoint, or datasets were installed/downloaded. Earlier verification tables describe their historical phases; this acquisition record supersedes their no-checkpoint status.

## First real pretrained CPU inference ? successful

Date: 2026-11-08. This record supersedes earlier historical statements that pretrained inference or model compatibility had not been verified.

### Input and attribution

- Local input: `data/images/coffee.png`, unmodified RGB image, **600 ? 400 pixels**.
- Content: coffee cup, spoon, saucer, and wooden tabletop; selected to exercise multiple objects and background regions.
- Photographer: **Rachel Michetti**, courtesy of Pikolo Espresso Bar.
- License: **CC0**, as stated by the [official scikit-image data documentation](https://scikit-image.org/docs/0.22.x/api/skimage.data.html#skimage.data.coffee).
- Exact download URL: https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/coffee.png
- Input SHA-256: `cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7`.

Image acquisition command (PowerShell, project root):

```powershell
New-Item -ItemType Directory -Path data/images -Force | Out-Null
curl.exe --fail --location --silent --show-error --output data/images/coffee.png https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/coffee.png
```

### Exact inference command

Run from the project root:

```text
docker compose run --rm --no-deps --pull never cpu python /workspace/scripts/infer_panoptic.py --input data/images/coffee.png --checkpoint checkpoints/coco_panoptic_eomt_small_640_2x/pytorch_model.bin --output-dir results/inference/first-pretrained-coffee
```

The output directory must not already exist; use a new output directory for a repeat run. No Docker rebuild was needed: the existing image supplies dependencies and Compose mounts the wrapper and upstream source read-only. Container networking was disabled by Compose.

| Measurement / setting | Observed value |
|---|---|
| Model | Official EoMT-S, DINOv2, COCO panoptic, 640 ? 640, 2x schedule, 200 queries, 133 classes plus no-object |
| Checkpoint | Original `pytorch_model.bin`; pinned revision and SHA-256 in acquisition record above; runtime hash matched |
| Source revision | `aa017c34afbe023de9462b17ae6067b52fef9867`; upstream working tree remained clean |
| Config SHA-256 | `264d06a721a592f260af21a9651ab83ba1145ba53ba7d7cb08a666b03320d77a` |
| Device / precision | CPU / float32 |
| Runtime | PyTorch `2.7.0+cpu`, Linux x86_64 under Docker Desktop / WSL2, 4 PyTorch intra-op threads |
| Input / output size | 600 ? 400 pixels |
| Model load duration | **10.115994 seconds** (includes ML imports, construction, local checkpoint loading) |
| Prediction duration | **1.651812 seconds** (preprocessing, forward, panoptic postprocessing, NumPy conversion; excludes loading and file saving) |
| Thresholds | Mask confidence 0.8; overlap 0.8 |
| Exit status | 0 |
| Predicted segments | 5 total: 3 thing instances and 2 stuff regions |
| Contiguous class IDs | 41, 44, 60, 121, 122; void 133 |
| Void pixels | 493 of 240,000 |

These timings describe one run, without warm-up or a throughput benchmark. No total process wall-clock measurement was recorded.

### Outputs and verification

All files are under `results/inference/first-pretrained-coffee/0000_coffee/`:

| File | Verified contents | Bytes |
|---|---|---|
| `segmentation.png` | Readable RGB class-color visualization with class/instance borders, 600 ? 400 | 4,119 |
| `overlay.png` | Readable RGB input/segmentation blend, 600 ? 400 | 365,941 |
| `prediction.npz` | Readable with `allow_pickle=False`; `semantic` and `segment_ids`, each int32 with shape (400, 600) | 8,710 |
| `prediction.json` | Readable JSON; model/config/checkpoint hashes, input hash, timing, dimensions, thresholds, segment IDs, class/category IDs, thing flags, pixel areas | 1,546 |

Validation confirmed class IDs in range 0?133, segment IDs ? -1, matching void class/segment masks, agreement between array segment IDs and JSON records, one class per segment, exact segment-area agreement, and populated COCO category IDs. Both PNGs were fully decoded. The overlay was visually inspected: cup and spoon regions are distinct; the saucer appears merged with the table region. Class colors are presentation aids. No ground-truth comparison or accuracy claim is made.

The wrapper needed no compatibility fix and inference succeeded on its first attempt. The only wrapper change added `time.perf_counter()` measurements (`model_load_seconds`, `prediction_seconds`) and `input_sha256` to saved metadata and printed timing. Model logic and upstream code were unchanged.

Regression verification after this change:

```text
docker compose run --rm --no-deps --pull never --workdir /workspace cpu python -m unittest tests.test_inference -v
```

Result: **10 tests passed, zero skipped**, 6.923 seconds. Non-fatal Lightning hyperparameter and Matplotlib deprecation warnings were observed across inference/tests. No training, fine-tuning, or quantitative dataset evaluation was performed. No additional model weights or dependencies were downloaded. Stopped after successful inference and output verification.
