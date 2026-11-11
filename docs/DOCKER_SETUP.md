# Docker setup and verification

## Scope and current status

Phase 3 adds a local-checkpoint inference wrapper and weight-free tests; see [INFERENCE.md](INFERENCE.md). Compose now mounts scripts/tests read-only and maps project-relative runtime paths. Dockerfiles include those files; dependency pins and default environment checks are unchanged. A successful CPU image rebuild and all 10 Phase 3 tests in Windows Docker were reported, with zero skips, including PyTorch integration tests. Matplotlib emitted non-fatal deprecation warnings. Pretrained inference and evaluation subsequently completed; see [EVALUATION.md](EVALUATION.md) and [FULL_MASKING_RESULTS.md](FULL_MASKING_RESULTS.md).

The initial setup phase added container definitions and environment smoke checks only. No dataset or pretrained checkpoint was downloaded, model instantiated, training started, or evaluation run.

CPU targets: Linux amd64 on Windows x86_64, Linux arm64 on Apple Silicon macOS, and Linux hosts. CUDA target: Linux amd64 with a supported NVIDIA GPU. **Windows x86_64 CPU Docker testing: PASSED**, based on the user's successful PowerShell build and smoke-test run. **macOS ARM64 and NVIDIA GPU testing: NOT VERIFIED.** Native Linux host testing is also not verified. Earlier Docker permission failures are recorded below.

The upstream source is unchanged at `aa017c34afbe023de9462b17ae6067b52fef9867`. Its Git working tree was clean before and after setup.

## Files and dependency decisions

| File | Purpose |
|---|---|
| `Dockerfile.cpu` | Native multi-architecture Python base, official CPU PyTorch wheels, dependencies, build-time smoke checks |
| `Dockerfile.cuda` | Same environment with CUDA 12.6 PyTorch wheels; GPU check deferred to container runtime |
| `compose.yaml` | Default CPU service and optional GPU profile; relative mounts, read-only source/data, offline runtime |
| `.dockerignore` | Allowlisted build context; excludes Git metadata, local runtime directories, common weight/archive files |
| `docker/requirements-torch.txt` | PyTorch 2.7.0 and torchvision 0.22.0 pins |
| `docker/requirements-common.txt` | Upstream direct package pins plus explicit pins for direct transitive imports |
| `docker/smoke_test.py` | Python/version, tensor, torchvision operator, and all project module import checks |
| `data/`, `checkpoints/`, `results/`, `cache/` | Local runtime directories; create them before first use |
| `.gitignore` | Excludes contents of runtime directories from future project commits |
| `README.md` | Entry point to setup and audit documents |

Both images use `python:3.11.12-slim-bookworm`. Python 3.11 replaces the README's 3.13.2 environment to use a mature Linux wheel ecosystem while preserving the upstream ML library versions. Debian Bookworm meets the selected manylinux wheel baseline. No host Python environment is installed or changed.

The [official PyTorch release instructions](https://pytorch.org/get-started/previous-versions/#v270) pair torch 2.7.0 with torchvision 0.22.0 and provide CPU/cu126 indexes. The [torch CPU wheel index](https://download.pytorch.org/whl/cpu/torch/) lists CPython 3.11 Linux x86_64 and aarch64 CPU wheels. The [torchvision CPU index](https://download.pytorch.org/whl/cpu/torchvision/) lists the matching x86_64 wheel with a +cpu suffix and the aarch64 wheel without that suffix. The requirement pins accept those local version differences.

CPU service has no forced platform: Docker builds for the native Linux architecture. Do not set `DOCKER_DEFAULT_PLATFORM=linux/amd64` on Apple Silicon unless deliberately testing emulation. CUDA service explicitly selects `linux/amd64`.

PyTorch is installed first using only its selected official index. Common dependencies then use normal PyPI with torch/torchvision constraints, preventing replacement with a different release. CUDA wheels carry user-space runtime libraries; the host supplies NVIDIA driver access. No CUDA compiler toolkit is included.

Upstream's direct pins are preserved. NumPy 1.26.4, Pillow 11.1.0, PyYAML 6.0.2, and huggingface-hub 0.34.4 are additionally pinned because the project imports them directly. Installer tooling is pinned to pip 25.1.1, setuptools 80.9.0, wheel 0.45.1. This is **not a full transitive lockfile** and the base tag is not digest-locked. Dependency installation and compatibility passed in the user's Windows x86_64 CPU image build and smoke-test run; ARM64 and CUDA dependency builds remain unverified. Builds run `pip check` before smoke checks and fail on errors.

No apt packages are currently needed by the intended wheel-based environment; source-only dependencies such as fvcore are Python packages. If a target requires a native build rather than an available wheel, capture the actual failure before adding compiler/system dependencies.

## Prerequisites

Install/start Docker Desktop in Linux-container mode on Windows or macOS. On Windows, use the WSL 2 backend. On Linux, use Docker Engine with Compose v2 or later. The CLI account must have access to the engine and its buildx state directory.

Check `docker version` includes a working Server section and `docker compose version` succeeds. Images and Python packages require build-time network access to Docker Hub, PyPI, and download.pytorch.org. Runtime networking is disabled by Compose.

For optional NVIDIA support, use compatible hardware and drivers. Windows needs Docker Desktop's WSL 2 backend with NVIDIA GPU passthrough; Linux needs NVIDIA Container Toolkit configured for Docker. See [Docker Desktop GPU prerequisites](https://docs.docker.com/desktop/features/gpu/) and [Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/). macOS uses the CPU service; this setup does not expose Apple MPS through Linux containers.

Obtain the pinned source first with `python scripts/obtain_eomt.py` from the project root. This requires Git and network access and refuses an existing destination. It was not executed during publication preparation. Create `data`, `checkpoints`, `results` and `cache` locally before using Compose. See the [project README](../README.md) for the complete setup sequence.

## Windows CMD

Run from the project root (the Dockerfiles and compose.yaml are here, not inside upstream):

```bat
cd /d "C:\path\to\EoMT-Panoptic-Segmentation"
docker version
docker compose version
docker compose config --quiet
docker compose build cpu
docker compose run --rm --no-deps cpu
```

Open a Linux shell inside the CPU environment if needed:

```bat
docker compose run --rm --no-deps cpu bash
```

## Windows PowerShell

```powershell
Set-Location 'C:\path\to\EoMT-Panoptic-Segmentation'
docker version
docker compose version
docker compose config --quiet
docker compose build cpu
docker compose run --rm --no-deps cpu
```

Stop and inspect the error if any command fails. Each command above is a separate invocation; no Bash line-continuation syntax is required.

## Apple Silicon macOS Terminal

Use the path where this project is stored on your Mac:

```sh
cd "/path/to/EoMT-Panoptic-Segmentation"
docker version
docker compose version
docker compose config --quiet
docker compose build cpu
docker compose run --rm --no-deps cpu
```

The CPU image should report `machine=aarch64` on a native Apple Silicon build and `machine=x86_64` on Windows x86_64. Both use Linux Python, not the host interpreter.

## Optional NVIDIA environment (Windows CMD, PowerShell, or Linux shell)

After the GPU prerequisites are configured, these single-line commands work in each shell:

```text
docker compose --profile gpu config --quiet
docker compose --profile gpu build cuda
docker compose --profile gpu run --rm --no-deps cuda
```

The GPU runtime test explicitly requires CUDA availability and performs a tiny CUDA tensor multiplication. Missing GPU access fails the test. Build-time checks run on CPU even for the CUDA image because GPUs are not normally available during image build. CUDA build/runtime tests have not been run here.

## Mounts and offline behavior

All host paths in Compose are relative to this project's compose.yaml:

| Host directory | Container directory | Access |
|---|---|---|
| `./upstream/eomt` | `/workspace/upstream/eomt` | Read-only |
| `./data` | `/data` | Read-only |
| `./checkpoints` | `/checkpoints` | Read-only |
| `./results` | `/results` | Read/write |
| `./cache` | `/cache` | Read/write |

Source is also copied into each image for standalone import checks. Compose mounts the local source read-only over that copy, so rebuild and rerun checks after intentionally changing upstream revisions. Bytecode writes are disabled. Data, weights, results, and cache contents are excluded from the build context and are not baked into images.

In a later phase, dataset/checkpoint paths should use `/data` and `/checkpoints`, and output/default-root paths should use `/results`. No training/evaluation command is included in this phase. Containers run as the base image's default root user; on native Linux, generated output files can therefore be root-owned.

Compose disables networking and sets Hugging Face offline flags and W&B offline mode. This prevents automatic checkpoint retrieval during checks. No W&B login is needed. Offline mode retains the W&B API expected by upstream rather than removing its logger. Cache directories are mounted but initially empty.

Matplotlib uses Agg with its configuration under /tmp; HF/Torch caches use /cache; W&B uses /results. Shared memory is set to 2 GB for future data-loader use, without asserting a tested memory requirement.

The upstream notebook still hardcodes CUDA. This phase does not change notebook code, mixed-precision defaults, compilation defaults, dataset handling, or evaluation behavior. Future CPU workflows will need explicit full precision/CPU selection and likely disabled compilation; defining a CPU image does not alone make the notebook CPU-compatible.

## Smoke checks

Both Dockerfiles run checks after installing dependencies, without constructing any model. The default Compose command repeats them against the mounted source, with runtime networking disabled.

Checks cover:

- Exact Python 3.11.12, torch 2.7.0, and torchvision 0.22.0 release versions.
- Machine architecture and CUDA build/availability reporting.
- CPU tensor multiplication and torchvision's native NMS operator (detects binary pairing problems).
- Imports of every Python module under upstream models, datasets, and training, then main without invoking its CLI.
- Import resolution points to the intended upstream source, avoiding unrelated packages named datasets.
- CPU image contains a non-CUDA torch build; optional GPU runtime can execute CUDA tensor operations.

Success prints `Environment smoke tests passed; no model or dataset was loaded.` A successful Windows PowerShell CPU smoke-test run was reported; it was not independently rerun for this record. The script does not access dataset archives, instantiate backbones, run a model forward pass, or calculate evaluation metrics.

After a successful build, dependency inventory can be saved without running a model:

```text
docker compose run --rm --no-deps cpu python -m pip freeze > results/environment-cpu.txt
```

## Verification record (2026-11-08)

### Phase 2: user-verified Windows CPU environment run

Evidence source: the user's report of commands executed successfully from Windows PowerShell. No raw build/test log or dependency inventory was supplied; this record does not represent an independent container test rerun.

```powershell
docker compose build cpu
docker compose run --rm --no-deps cpu
```

| Platform / environment | Status |
|---|---|
| Windows x86_64, Linux CPU container | **PASSED** |
| Apple Silicon macOS ARM64, Linux CPU container | **NOT VERIFIED** |
| NVIDIA CUDA GPU, Windows/Linux amd64 | **NOT VERIFIED** |
| Native Linux host CPU | **NOT VERIFIED** |

Reported results:

- CPU image built successfully.
- Python 3.11.12, PyTorch 2.7.0+cpu, and torchvision 0.22.0+cpu.
- All EoMT model, dataset, and training imports passed.
- Docker container started and exited successfully.
- No model, checkpoint, or dataset was loaded.

The reported successful run supersedes the earlier blocked Windows CPU verification status. These environment checks do not establish model inference, training, evaluation, ARM64, or GPU compatibility and did not load checkpoints or execute a model.

### Additional verification: Phase 3 Windows CPU inference tests

The user reported successful execution of:

```text
docker compose build cpu
docker compose run --rm --no-deps --workdir /workspace cpu python -m unittest tests.test_inference -v
```

Result: **PASSED — CPU image built; all 10 tests passed with zero skips, including PyTorch integration tests.** Matplotlib produced non-fatal deprecation warnings. This is weight-free workflow verification, not pretrained model inference. macOS ARM64 and NVIDIA GPU remain **NOT VERIFIED**. These weight-free checks did not download checkpoints or datasets or start inference or training. See [Phase 3 verification details](INFERENCE.md#verification-results).

### Earlier Docker access failures (Phase 2)

Host: Windows amd64. Docker client 29.9.2, Compose v5.6.1, Docker context desktop-linux. Host Python 3.13 was used only to parse the smoke script's syntax, not to run container imports.

| Command/check actually performed | Result |
|---|---|
| `docker version` | Client available; server access failed with permission denied |
| `docker compose version` | Passed: v5.6.1 |
| `docker compose config --quiet` | Passed, exit 0 |
| `docker compose --profile gpu config --quiet` | Passed, exit 0 |
| Host `ast.parse` of docker/smoke_test.py | Passed syntax parsing only |
| `docker compose build cpu` | Failed before image/dependency build, exit 1 |
| `docker compose run --rm --no-deps cpu` | Failed before container execution, exit 1 |
| Upstream HEAD and `git status --porcelain` | Expected commit; clean working tree |

Exact CPU build blocker:

```text
Image eomt-reproduction:cpu Building
open <user-profile>/.docker/buildx/.lock: Access is denied.
```

Exact engine/runtime blocker:

```text
permission denied while trying to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

The earlier checks failed because buildx state and the Docker Desktop engine were inaccessible. Docker settings and network policy were unchanged. The subsequent successful PowerShell build and runtime checks are reported above.

The failed attempts stopped before dependency installation or container execution, so they provide no evidence of dependency or network incompatibility. They are distinct from the subsequent reported successful Windows CPU run. ARM64 and GPU runtime tests remain **NOT VERIFIED**.

Remaining verification: build/run the CPU service on native Apple Silicon arm64, then optionally build/run CUDA on a compatible NVIDIA host. Save raw build/smoke output and dependency inventories for reproducibility when available. Windows x86_64 CPU testing is **PASSED**. Model inference, training, and evaluation belong to later phases.

