"""Offline CPU panoptic inference using the unmodified official EoMT classes."""
# Visualization boundary logic follows upstream/eomt/inference.ipynb (MIT):
# Copyright (c) 2025 Mobile Perception Systems Lab at TU/e.
# The upstream permission notice is retained in upstream/eomt/LICENSE.
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "eomt"
COMMIT = "aa017c34afbe023de9462b17ae6067b52fef9867"
MODEL_ID = "tue-mps/coco_panoptic_eomt_small_640_2x"
CHECKPOINT_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main/pytorch_model.bin"
CONFIG = UPSTREAM / "configs/dinov2/coco/panoptic/eomt_small_640_2x.yaml"
DEFAULT_CHECKPOINT = "checkpoints/coco_panoptic_eomt_small_640_2x/pytorch_model.bin"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def project_path(value: str | Path) -> Path:
    """Resolve against project root, translating Compose's runtime mounts."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] in {"data", "checkpoints", "results"}:
        mounted = os.environ.get(f"EOMT_{path.parts[0].upper()}_DIR")
        if mounted:
            return Path(mounted).joinpath(*path.parts[1:]).resolve()
    return (ROOT / path).resolve()


def require_checkpoint(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Expected the original upstream-format pytorch_model.bin for {MODEL_ID}.\n"
            f"Official source: {CHECKPOINT_URL}\n"
            "No download was attempted. Place the checkpoint locally when authorized, "
            "or pass --checkpoint with its existing path."
        )
    if path.stat().st_size == 0:
        raise ValueError(f"Checkpoint is empty: {path}")
    return path


def collect_images(values: list[str]) -> list[Path]:
    images = []
    for value in values:
        path = project_path(value)
        if path.is_dir():
            found = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if not found:
                raise ValueError(f"No supported images in directory: {path}")
            images.extend(found)
        elif path.is_file() and path.suffix.lower() in EXTENSIONS:
            images.append(path)
        else:
            raise ValueError(f"Input image does not exist or has an unsupported extension: {path}")
    return list(dict.fromkeys(images))


def offline_environment() -> None:
    # Set before importing timm/Transformers; no Hub APIs are used by this script.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    if not (UPSTREAM / "models/vit.py").is_file():
        raise FileNotFoundError(f"Official EoMT checkout missing: {UPSTREAM}")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))


def load_model(checkpoint: Path):
    """Build the official small 2x configuration and load local absolute weights."""
    require_checkpoint(checkpoint)
    offline_environment()
    import torch
    import yaml
    from lightning import seed_everything
    from models.eomt import EoMT
    from models.vit import ViT
    from training.mask_classification_panoptic import MaskClassificationPanoptic

    seed_everything(0, verbose=False)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    model_args = dict(config["model"]["init_args"])
    network_args = dict(model_args.pop("network")["init_args"])
    encoder_args = dict(network_args.pop("encoder")["init_args"])
    data_args = config["data"]["init_args"]
    # Match defaults from COCOPanoptic, without creating or scanning a dataset.
    img_size, num_classes = (640, 640), 133
    # A non-None ckpt_path makes official ViT call timm with pretrained=False.
    # All encoder parameters subsequently come from the EoMT checkpoint.
    encoder = ViT(img_size=img_size, ckpt_path=str(checkpoint), **encoder_args)
    network = EoMT(encoder=encoder, num_classes=num_classes,
                   masked_attn_enabled=False, **network_args)
    try:
        model = MaskClassificationPanoptic(
            network=network, img_size=img_size, num_classes=num_classes,
            stuff_classes=data_args["stuff_classes"], ckpt_path=str(checkpoint),
            delta_weights=False, load_ckpt_class_head=True, **model_args,
        )
    except (ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError(
            f"Cannot load checkpoint {checkpoint}: {error}. "
            f"Use the original upstream-format pytorch_model.bin for {MODEL_ID}; "
            "Transformers-converted weights and backbone-only weights are incompatible."
        ) from error
    # Upstream loads map_location=cpu, weights_only=True and rejects incompatible
    # missing/unexpected model keys. Never continue with partial/random weights.
    return model.to(device=torch.device("cpu"), dtype=torch.float32).eval()


def predict_image(model, image):
    """Return H×W×2 class/segment IDs using official panoptic helpers."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    img = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.uint8, copy=True)).permute(2, 0, 1)
    with torch.inference_mode():
        transformed = model.resize_and_pad_imgs_instance_panoptic([img])
        # Official Lightning forward divides uint8 values by 255; EoMT applies
        # ImageNet normalization internally. Do not normalize a second time.
        mask_layers, class_layers = model(transformed)
        masks = F.interpolate(mask_layers[-1], model.img_size, mode="bilinear")
        masks = model.revert_resize_and_pad_logits_instance_panoptic(masks, [img.shape[-2:]])
        preds = model.to_per_pixel_preds_panoptic(
            masks, class_layers[-1], model.stuff_classes,
            model.mask_thresh, model.overlap_thresh,
        )[0]
    return preds.cpu().numpy().astype(np.int32)


def render_prediction(prediction, void_class=133):
    """Notebook-style class colors and black class/instance borders."""
    import numpy as np

    semantic, segments = prediction[..., 0], prediction[..., 1]
    ids = np.unique(semantic)
    colors = {int(c): colorsys.hsv_to_rgb(i / len(ids), 1, 1) for i, c in enumerate(ids)}
    colors[void_class] = colors[-1] = (0, 0, 0)
    canvas = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    for cls in ids:
        canvas[semantic == cls] = np.round(np.asarray(colors[int(cls)]) * 255).astype(np.uint8)
    borders = np.zeros(semantic.shape, dtype=bool)
    vertical = (semantic[1:] != semantic[:-1]) | (segments[1:] != segments[:-1])
    horizontal = (semantic[:, 1:] != semantic[:, :-1]) | (segments[:, 1:] != segments[:, :-1])
    borders[1:] |= vertical
    borders[:-1] |= vertical
    borders[:, 1:] |= horizontal
    borders[:, :-1] |= horizontal
    canvas[borders] = 0
    return canvas


def save_prediction(image, prediction, destination: Path, metadata: dict) -> None:
    import numpy as np
    from PIL import Image

    semantic, segments = prediction[..., 0], prediction[..., 1]
    if semantic.shape != (image.height, image.width):
        raise ValueError("Prediction dimensions do not match the input image")
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / "prediction.npz", semantic=semantic, segment_ids=segments)
    colored = Image.fromarray(render_prediction(prediction))
    colored.save(destination / "segmentation.png")
    overlay = Image.blend(image.convert("RGB"), colored, 0.5)
    overlay.save(destination / "overlay.png")
    metadata = dict(metadata)
    from_mapping = metadata.pop("category_mapping", {})
    segment_info = []
    for segment_id in np.unique(segments):
        if segment_id < 0:
            continue
        mask = segments == segment_id
        cls = int(semantic[mask][0])
        segment_info.append({
            "id": int(segment_id), "class_id": cls,
            "coco_category_id": from_mapping.get(cls), "isthing": cls < 80,
            "area": int(mask.sum()),
        })
    record = {**metadata, "width": image.width, "height": image.height,
              "void_class_id": 133, "void_segment_id": -1, "segments_info": segment_info}
    (destination / "prediction.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Local image files or nonrecursive directories")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Local upstream-format checkpoint; never downloaded")
    parser.add_argument("--output-dir", default="results/inference", help="New run directory underneath results/")
    args = parser.parse_args(argv)
    try:
        checkpoint = require_checkpoint(project_path(args.checkpoint))
        images = collect_images(args.input)
        output = project_path(args.output_dir)
        if not output.is_relative_to(project_path("results")):
            raise ValueError("--output-dir must be underneath the project's results/ directory")
        if output.exists():
            raise ValueError(f"Output directory already exists: {output}. Choose a new --output-dir to avoid overwriting predictions.")
        load_started = time.perf_counter()
        model = load_model(checkpoint)
        model_load_seconds = time.perf_counter() - load_started
        from PIL import Image
        from datasets.coco_panoptic import CLASS_MAPPING

        provenance = {
            "model_id": MODEL_ID, "checkpoint_source": CHECKPOINT_URL,
            "checkpoint_sha256": file_sha256(checkpoint), "source_commit_expected": COMMIT,
            "config": CONFIG.relative_to(ROOT).as_posix(),
            "config_sha256": file_sha256(CONFIG), "device": "cpu", "precision": "float32",
            "model_load_seconds": model_load_seconds,
            "mask_threshold": model.mask_thresh, "overlap_threshold": model.overlap_thresh,
            "category_mapping": {v: k for k, v in CLASS_MAPPING.items()},
        }
        for index, path in enumerate(images):
            with Image.open(path) as source:
                image = source.convert("RGB")
            prediction_started = time.perf_counter()
            prediction = predict_image(model, image)
            prediction_seconds = time.perf_counter() - prediction_started
            destination = output / f"{index:04d}_{path.stem}"
            save_prediction(image, prediction, destination, {**provenance, "input": str(path),
                            "input_sha256": file_sha256(path),
                            "prediction_seconds": prediction_seconds})
            print(f"Saved {path.name}: {destination} (prediction {prediction_seconds:.3f}s; model load {model_load_seconds:.3f}s)")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Inference error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
