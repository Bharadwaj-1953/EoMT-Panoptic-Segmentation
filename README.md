<h1 align="center">
EoMT Panoptic Segmentation Reproduction and <br>
Ablation Analysis
</h1>

---

## 📝 Abstract

<div align="justify">

This project evaluates the official pretrained Encoder-only Mask Transformer (EoMT) for COCO panoptic segmentation and examines its sensitivity to inference-time changes in query count, query insertion depth and attention masking. It uses the released DINOv2 EoMT-S 640 × 640 checkpoint trained with the authors' 24-epoch 2x schedule. Evaluation on all 5,000 COCO 2017 validation images obtains 46.7364 Panoptic Quality (PQ), consistent with the official 46.7 PQ checkpoint benchmark at its published precision.

The reproduction covers pretrained-checkpoint evaluation and fixed-checkpoint inference ablations. Model training and fine-tuning were not independently reproduced. Five configurations were compared on the same seeded 500-image subset, followed by a full-validation masking experiment. Saved predictions passed separate-process verification using the vendored COCO reference evaluator. Full attention masking obtains 45.6785 PQ, a decrease of 1.0579 points from the full-validation baseline.

</div>

---

## 📁 Repository Structure

Selected publication files are shown below. Local datasets, checkpoints, the separately acquired upstream checkout and generated experiment outputs are excluded.

```text
EoMT-Panoptic-Segmentation/
├── docker/
│   ├── requirements-common.txt
│   ├── requirements-torch.txt
│   └── smoke_test.py
├── docs/
│   ├── DOCKER_SETUP.md
│   ├── INFERENCE.md
│   ├── EVALUATION.md
│   ├── ABLATIONS.md
│   ├── EXPERIMENT_ANALYSIS.md
│   ├── PAPER_COMPARISON.md
│   └── FULL_MASKING_RESULTS.md
├── licenses/
│   └── EoMT-MIT.txt
├── publication/
│   └── results/
│       ├── verified_metrics.csv
│       ├── masking_comparison.csv
│       └── provenance.json
├── scripts/
│   ├── obtain_eomt.py
│   ├── infer_panoptic.py
│   ├── evaluate_panoptic.py
│   ├── ablation_interventions.py
│   ├── evaluate_ablations.py
│   ├── run_ablation_suite.py
│   ├── verify_ablation.py
│   ├── smoke_ablations.py
│   ├── analysis/
│   │   ├── analyze_saved.py
│   │   ├── refine_exports.py
│   │   ├── finalize_analysis.py
│   │   └── compare_verified.py
│   ├── verification/
│   │   ├── check_inputs.py
│   │   ├── audit_saved.py
│   │   └── verify_and_rescore.py
│   └── vendor/
│       ├── README.md
│       └── panopticapi/
│           ├── evaluation.py
│           ├── utils.py
│           └── LICENSE
├── tests/
│   ├── test_inference.py
│   ├── test_evaluation.py
│   └── test_ablations.py
├── .dockerignore
├── .gitignore
├── compose.yaml
├── Dockerfile.cpu
├── Dockerfile.cuda
├── THIRD_PARTY_NOTICES.md
└── README.md
```

The tree omits package markers. The entire local report directory and internal preparation, audit and planning documents are excluded from the initial publication.

---

## 🎯 Key Features

- Offline CPU inference through the official EoMT model classes, with local checkpoint loading.
- Full or first-N COCO validation evaluation with PQ, Segmentation Quality (SQ), Recognition Quality (RQ), and per-class results.
- Seeded, matched-image interventions on learned queries, final joint blocks and predicted attention restrictions.
- Sequential five-arm execution with independent saved-mask verification after each arm.
- SHA-256 provenance for checkpoints, configurations, inputs and selected result summaries.
- Reusable saved-artifact analysis tools, with historical experiment records preserved separately.

---

## 🛠️ Technologies Used

| Component | Technology |
|---|---|
| Container runtime | Docker and Docker Compose v2 |
| Container Python | Python 3.11.12 |
| Tensor libraries | PyTorch 2.7.0, torchvision 0.22.0 |
| Model implementation | Official EoMT, DINOv2 backbone, timm 1.0.15 |
| Model infrastructure | Lightning 2.5.1.post0, Transformers 4.56.1 |
| Data and analysis | NumPy 1.26.4, Pillow 11.1.0, SciPy 1.15.2, Matplotlib 3.10.1 |
| Evaluation | Vendored `cocodataset/panopticapi` reference scorer |

These are container dependency pins. The historical host-side analysis used a different environment, recorded in [EXPERIMENT_ANALYSIS.md](docs/EXPERIMENT_ANALYSIS.md). Transitive dependencies and container base-image digests are not fully locked.

---

## ⚙️ Setup and Installation

### 1. Prerequisites and project checkout

Install Git, host Python 3.11 or newer, and Docker with Compose v2. Use Linux containers; Windows Docker Desktop uses the WSL 2 backend. A GPU is not required. Allow disk space for dataset archives, extracted images and masks, the Docker image, and new prediction outputs.

Once the repository is published at its intended GitHub address, clone it with:

```bash
git clone https://github.com/Bharadwaj-1953/EoMT-Panoptic-Segmentation.git
cd EoMT-Panoptic-Segmentation
```

Run all subsequent commands from this project root. Commands use `python` for the host interpreter; use `python3` if that is its installed name. Single-line Python and Docker commands also work in PowerShell.

### 2. Obtain the pinned official source

```bash
python scripts/obtain_eomt.py
```

The helper clones [the official EoMT repository](https://github.com/tue-mps/eomt), checks out detached commit `aa017c34afbe023de9462b17ae6067b52fef9867` into `upstream/eomt`, and verifies HEAD. It requires network access and refuses an existing destination. For an existing checkout, inspect its revision without replacing it:

```bash
git -C upstream/eomt rev-parse HEAD
```

### 3. Prepare Docker

Create the local runtime directories, then build and check the CPU environment:

```bash
python -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ('data/downloads', 'checkpoints/coco_panoptic_eomt_small_640_2x', 'results', 'cache')]"
docker version
docker compose version
docker compose config --quiet
docker compose build cpu
docker compose run --rm --no-deps cpu
```

The final command checks the environment; it does not run pretrained inference. Builds require network access for dependencies. Compose disables runtime networking and mounts source, data and checkpoints read-only. See [DOCKER_SETUP.md](docs/DOCKER_SETUP.md) for platform details.

Windows x86_64 CPU execution is documented. Apple Silicon ARM64 and native Linux host execution remain unverified. An optional Linux amd64 CUDA 12.6 environment requires a supported NVIDIA GPU, driver and Docker GPU integration:

```bash
docker compose --profile gpu build cuda
docker compose --profile gpu run --rm --no-deps cuda
```

CUDA build/runtime validation remains unverified. The project inference, evaluation and ablation CLIs implement CPU float32 execution; selecting the CUDA service does not add GPU inference support. Linux containers on macOS do not expose Apple MPS through this setup.

### 4. Obtain and verify the checkpoint

Use the original `pytorch_model.bin` from [tue-mps/coco_panoptic_eomt_small_640_2x](https://huggingface.co/tue-mps/coco_panoptic_eomt_small_640_2x), pinned to Hub revision `10f5326794a8ce7e0e975791033ae1c13aec3767`. The following URL and checksum are recorded in the [checkpoint acquisition record](docs/INFERENCE.md#official-checkpoint-acquisition-and-verification):

```bash
python -c "from urllib.request import urlretrieve; urlretrieve('https://huggingface.co/tue-mps/coco_panoptic_eomt_small_640_2x/resolve/10f5326794a8ce7e0e975791033ae1c13aec3767/pytorch_model.bin', 'checkpoints/coco_panoptic_eomt_small_640_2x/pytorch_model.bin')"
python -c "import hashlib; from pathlib import Path; p = Path('checkpoints/coco_panoptic_eomt_small_640_2x/pytorch_model.bin'); digest = hashlib.sha256(p.read_bytes()).hexdigest(); print(digest); assert digest == '7365b41a8a129afc74c12e5dd7af534146286d02822e4b89b7bbe8ea613ff74d', 'Checkpoint SHA-256 mismatch'"
```

The verified file is 95,968,930 bytes. Do not substitute the Transformers-converted `model.safetensors`. Skip the download when the correct checkpoint already exists. Checkpoint and dataset acquisition commands below are for fresh installations and can overwrite existing destination files. Download availability was established in the historical records and was not rechecked during README preparation.

### 5. Obtain COCO 2017 validation data

Obtain the validation JPEG archive and panoptic annotation archive from the [official COCO download listing](https://cocodataset.org/#download). The URLs below are the sources used in the recorded acquisition:

```bash
python -c "from urllib.request import urlretrieve; urlretrieve('http://images.cocodataset.org/zips/val2017.zip', 'data/downloads/val2017.zip')"
python -c "from urllib.request import urlretrieve; urlretrieve('http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip', 'data/downloads/panoptic_annotations_trainval2017.zip')"
python -m zipfile -e data/downloads/val2017.zip data/coco
python -m zipfile -e data/downloads/panoptic_annotations_trainval2017.zip data/coco
python -m zipfile -e data/coco/annotations/panoptic_val2017.zip data/coco/annotations
```

The final command extracts the nested validation-mask archive. The evaluator requires these local paths:

| Local path | Required content |
|---|---|
| `data/coco/val2017/` | 5,000 validation JPEGs with original filenames |
| `data/coco/annotations/panoptic_val2017.json` | Official image, segment and 133-category metadata |
| `data/coco/annotations/panoptic_val2017/` | 5,000 RGB segment-ID ground-truth PNGs |

Training images and the separate detection/instance annotation archive are unnecessary. Do not replace segment-ID masks with class-color visualizations. Dataset fingerprints and extraction details are recorded in [EVALUATION.md](docs/EVALUATION.md). Review COCO terms and image-specific rights before reuse or redistribution.

---

## 🚀 Running the Project

Use a new output directory for every run. The examples below assume the checkpoint and extracted dataset are prepared. Relative `data/`, `checkpoints/` and `results/` paths map to their Compose mounts, even though the container working directory is `/workspace/upstream/eomt`.

### Inference

Segment one image from the prepared validation split:

```bash
docker compose run --rm --no-deps cpu python /workspace/scripts/infer_panoptic.py --input data/coco/val2017/000000000139.jpg --output-dir results/inference/readme-run-001
```

`--input` also accepts local image paths or a nonrecursive image directory. Each image produces `prediction.npz`, `prediction.json`, `segmentation.png` and `overlay.png` in its own subdirectory. The checkpoint defaults to the verified path above. See [INFERENCE.md](docs/INFERENCE.md) for output semantics.

### Evaluation

Check the first ten images in ascending image-ID order:

```bash
docker compose run --rm --no-deps cpu python /workspace/scripts/evaluate_panoptic.py --device cpu --limit 10 --output-dir results/evaluation/readme-first10
```

Evaluate all 5,000 validation images:

```bash
docker compose run --rm --no-deps cpu python /workspace/scripts/evaluate_panoptic.py --device cpu --full --output-dir results/evaluation/readme-full
```

The default image, annotation, mask and checkpoint paths match the setup table. `--images`, `--panoptic`, `--annotations` and `--checkpoint` accept alternate local paths. Evaluation writes `metrics.json`, aggregate/per-class CSVs, timings, prediction metadata and panoptic prediction masks. Raw evaluation JSON/CSV scores use the 0–1 scale; multiply by 100 for metric points.

### Matched 500-image ablation suite

```bash
docker compose run --rm --no-deps cpu python /workspace/scripts/run_ablation_suite.py --output-dir results/ablations/readme-val500
```

The suite runs B0, Q100, L2, M1 and M0 sequentially in separate processes. Each arm uses the same seed-0 sample of 500 images and four intra-op threads, then undergoes independent saved-mask verification. It writes `suite.json`, `comparison.csv`, `comparison.json`, logs and per-arm artifacts.

Historical baseline evaluation, verification and smoke-test outputs are optional. Add `--check-historical-integrity` only when those original records are available; requested checks remain mandatory and compare their before/after hashes. Checkpoint identity is always checked. See [ABLATIONS.md](docs/ABLATIONS.md) for individual-arm settings.

### Full-validation attention masking

```bash
docker compose run --rm --no-deps cpu python /workspace/scripts/evaluate_ablations.py --attention full --sample-size 5000 --sample-seed 0 --threads 4 --output-dir results/ablations/readme-mask-full-5000
```

This runs the full M1 intervention. It does not invoke the 500-image suite verifier, which is specific to that subset. The historical full-run verification procedure and its limitations are documented in [FULL_MASKING_RESULTS.md](docs/FULL_MASKING_RESULTS.md).

### Access saved results

The compact, verified numerical summaries are available without datasets or model execution:

| Published file | Contents |
|---|---|
| [verified_metrics.csv](publication/results/verified_metrics.csv) | Overall, Things and Stuff PQ/SQ/RQ for all seven verified runs |
| [masking_comparison.csv](publication/results/masking_comparison.csv) | Full-validation and matched-pilot B0/M1 differences |
| [provenance.json](publication/results/provenance.json) | Revisions, checkpoint identity, source fingerprints and interpretation limits |

```bash
python -c "from pathlib import Path; print(Path('publication/results/verified_metrics.csv').read_text())"
```

Historical masks, per-image outputs and COCO-derived figures are not bundled. Saved-artifact analysis tools require those local inputs; their usage is described in [EXPERIMENT_ANALYSIS.md](docs/EXPERIMENT_ANALYSIS.md#maintained-analysis-commands).

---

## 📊 Experimental Results and Analysis

### Full COCO validation: 5,000 images

All values are 0–100 metric points. Both local runs use CPU float32, the same official pretrained checkpoint, 200 queries, three final joint blocks, and confidence/overlap thresholds of 0.8/0.8.

| Configuration | PQ | SQ | RQ | PQ difference from local B0 |
|---|---:|---:|---:|---:|
| Official EoMT-S 640 2x benchmark, 24 epochs | 46.7 | Not reported | Not reported | Not applicable |
| B0: unmasked baseline | 46.7364 | 81.5330 | 56.3355 | 0.0000 |
| M1: full predicted attention masking | 45.6785 | 81.2838 | 54.8686 | -1.0579 |

The local baseline rounds to the official benchmark at one decimal. The +0.0364-point difference from the displayed 46.7 is not evidence of an improvement over an unrounded author score. The paper's standard EoMT-S result is 44.7 PQ, associated with the 12-epoch configuration; it is a different reference from the released 24-epoch 2x checkpoint. See the [schedule and scorer comparison](docs/PAPER_COMPARISON.md) and [pinned official model zoo](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/model_zoo/dinov2.md).

### Five matched configurations: 500 images

Images are sampled without replacement from sorted validation IDs using `numpy.default_rng(0)`, then sorted again. Q100 uses a separate seed-0 selection of learned query embeddings. Only one factor changes per intervention.

| Arm | Queries | Final joint blocks | Attention setting | PQ | PQ difference from matched B0 |
|---|---:|---:|---|---:|---:|
| B0 | 200 | 3 | Masking off | 47.7820 | 0.0000 |
| Q100 | 100 | 3 | Masking off | 40.4920 | -7.2900 |
| L2 | 200 | 2 | Masking off | 28.8466 | -18.9355 |
| M1 | 200 | 3 | Predicted restrictions retained, probability 1 | 46.4318 | -1.3502 |
| M0 | 200 | 3 | Mask branch active, restrictions released, probability 0 | 47.7820 | 0.0000 |

Later query insertion produces the largest observed loss. Removing half the queries particularly reduces Stuff PQ. M0 retains byte-identical predictions to B0 for all 500 images while still computing intermediate heads and masks. Full masking has a negative overall effect in both the pilot and full validation runs. Detailed class and timing analysis is in [EXPERIMENT_ANALYSIS.md](docs/EXPERIMENT_ANALYSIS.md).

### Interpretation and verification limits

These are interventions on one pretrained checkpoint, not separately trained architectures. L2 retains all backbone blocks and changes where queries enter; M1 reactivates predicted attention restrictions at inference and does not reproduce training-time mask annealing. The pilot averages 131 active categories and the full runs average 133. The pilot is contained in the full split, so these are overlapping populations without confidence intervals or significance claims.

Separate-process rescoring of saved predictions agreed within `1e-12` for aggregate metrics using the same COCO reference implementation. This establishes saved-output scoring integrity, not equivalence to the authors' internal scorer. CPU timings are uncontrolled, and historical runs include a laptop-sleep clock anomaly; no throughput reproduction is claimed. The final audit also records incomplete host test-suite revalidation and unavailable container exit/log inspection. See [FULL_MASKING_RESULTS.md](docs/FULL_MASKING_RESULTS.md) for the verification evidence and remaining limits.

---

## 🔧 Troubleshooting

| Problem | Check or action |
|---|---|
| Source helper refuses the destination | Inspect the existing checkout and compare HEAD with the pinned commit; the helper intentionally does not replace it. |
| Docker daemon or engine-pipe error | Start Docker, select Linux containers and confirm `docker version` includes a Server section. On Windows, check the WSL 2 backend. |
| Missing or incompatible checkpoint | Use the original `pytorch_model.bin`, correct local path and recorded SHA-256. Runtime commands never download weights. |
| Missing ground-truth masks | Extract the nested `panoptic_val2017.zip`; confirm the PNG directory matches the setup table. |
| Output directory already exists | Choose a new run directory. Existing predictions are deliberately protected from overwrite. |
| Suite fails on historical integrity files | Omit `--check-historical-integrity` on fresh installations; use it only with the required historical records. |
| CUDA unavailable | Use the CPU service. The optional CUDA smoke check requires NVIDIA passthrough, and the project CLIs remain CPU-only. |
| Scores differ by a factor of 100 | Evaluation artifacts store 0–1 scores; published summaries and README tables use 0–100 points. |

Platform-specific setup details are in [DOCKER_SETUP.md](docs/DOCKER_SETUP.md). Earlier documentation retains dated phase records; use the final audit and verified summaries for completed experiment status.

---

## 📚 References and Attribution

- Kerssies, Tommie; Cavagnero, Niccolò; Hermans, Alexander; Norouzi, Narges; Averta, Giuseppe; Leibe, Bastian; Dubbelman, Gijs; and de Geus, Daan. [*Your ViT is Secretly an Image Segmentation Model*](https://arxiv.org/abs/2503.19108), CVPR 2025. Original EoMT research and implementation by the authors and the Mobile Perception Systems Lab at TU/e.
- [Official EoMT source](https://github.com/tue-mps/eomt), pinned to `aa017c34afbe023de9462b17ae6067b52fef9867`. Its MIT notice is preserved in [licenses/EoMT-MIT.txt](licenses/EoMT-MIT.txt).
- [DINOv2](https://github.com/facebookresearch/dinov2), by Meta AI Research/FAIR, supplies the pretrained visual backbone used by the released EoMT model. Refer to its own Apache-2.0 license and model documentation.
- [COCO](https://cocodataset.org/), by the COCO Consortium, supplies the validation data and panoptic task. Annotations are covered by CC BY 4.0; images retain individual owners' rights and applicable image-specific terms. See [COCO terms of use](https://cocodataset.org/#termsofuse).
- [COCO panoptic evaluator](https://github.com/cocodataset/panopticapi/tree/7bb4655548f98f3fedc07bf37e9040a992b054b0), copyright 2018 Alexander Kirillov, is vendored at that revision under its [BSD 2-clause license](scripts/vendor/panopticapi/LICENSE).

This repository contributes reproduction wrappers, inference interventions, verification and analysis around the upstream implementation. It does not claim ownership of the original research code. A project-wide license for original contributions has not yet been selected. Dataset, checkpoint, backbone and report redistribution require separate consideration; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## 📬 Contact

- GitHub: [Bharadwaj-1953](https://github.com/Bharadwaj-1953)
- LinkedIn: [Bharadwaj Manne](https://www.linkedin.com/in/bharadwaj-manne-711476249/)
- Portfolio: [bharadwaj.vercel.app](https://bharadwaj.vercel.app/)
- Email: [manne.bharadwaj.1953@gmail.com](mailto:manne.bharadwaj.1953@gmail.com)
