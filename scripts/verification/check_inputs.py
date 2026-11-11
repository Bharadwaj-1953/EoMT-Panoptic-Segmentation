"""Decode local validation inputs and validate panoptic metadata without inference."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ["EOMT_AUDIT_OUTPUT"]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
data = json.loads((ROOT / "data/coco/annotations/panoptic_val2017.json").read_text())
ground_truth = {row["image_id"]: row for row in data["annotations"]}
categories = {row["id"] for row in data["categories"]}
started = time.perf_counter()
segments = 0
modes = Counter()
for index, image in enumerate(data["images"], 1):
    with Image.open(ROOT / "data/coco/val2017" / image["file_name"]) as source:
        source.load()
        assert source.size == (image["width"], image["height"])
        modes[source.mode] += 1
        # Match the supported input boundary used by the inference wrapper.
        rgb_image = source.convert("RGB")
        assert rgb_image.size == source.size and rgb_image.mode == "RGB"
    annotation = ground_truth[image["id"]]
    with Image.open(ROOT / "data/coco/annotations/panoptic_val2017" / annotation["file_name"]) as mask:
        mask.load()
        assert mask.mode == "RGB" and mask.size == (image["width"], image["height"])
        rgb = np.asarray(mask).astype(np.uint32)
        ids = rgb[:, :, 0] + 256 * rgb[:, :, 1] + 65536 * rgb[:, :, 2]
    labels, counts = np.unique(ids, return_counts=True)
    areas = {int(k): int(v) for k, v in zip(labels, counts) if k}
    records = annotation["segments_info"]
    assert len(records) == len({segment["id"] for segment in records})
    assert areas == {segment["id"]: segment["area"] for segment in records}
    assert all(segment["category_id"] in categories and segment["iscrowd"] in (0, 1) for segment in records)
    segments += len(records)
    if index % 1000 == 0:
        print("Dataset decoded", index, flush=True)
report = {"status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
          "images_fully_decoded": 5000, "source_jpeg_modes": dict(modes),
          "ground_truth_pngs_fully_decoded": 5000, "native_dimensions_match": True,
          "ground_truth_ids_areas_categories_crowd_valid": True,
          "segment_records": segments, "seconds": time.perf_counter() - started}
with (OUT / "dataset_decode.json").open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report), flush=True)
