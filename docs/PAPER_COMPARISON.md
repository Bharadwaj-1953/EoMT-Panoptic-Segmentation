# EoMT-S 640 2x: comparison with the paper and official model zoo

Prepared 2026-11-09. This is a read-only review of published sources, source code,
and existing run artifacts. Only this document was added. No inference,
evaluation, rescoring, or training was run; no datasets or checkpoints were
downloaded; upstream source and existing results were not modified.

## Finding and correct published reference

**The exact DINOv2 EoMT-S 640 x 640 2x configuration has an official model-zoo
result of 46.7 COCO panoptic PQ. Our completed 5,000-image evaluation obtained
46.7364 PQ, agreeing at the published one-decimal precision.** The zoo row links
both the small 2x configuration and the original checkpoint repository. Its 2x
footnote identifies a longer training schedule. [Official DINOv2 model zoo](https://github.com/tue-mps/eomt/blob/master/model_zoo/dinov2.md#coco);
[pinned model zoo](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/model_zoo/dinov2.md).

The authors' paper, *Your ViT is Secretly an Image Segmentation Model*, Table 3,
reports **44.7 PQ for standard EoMT-S**, without masked attention, on COCO
val2017. That is a different schedule from the released 2x checkpoint; the
inspected paper does not provide a separate small 2x row. Appendix A.3 specifies
long-side resizing and square zero padding for panoptic inference.
[Author-hosted arXiv paper, Table 3 and Appendix A.3](https://arxiv.org/html/2503.19108v1).
The local standard and 2x configs specify 12 and 24 epochs, respectively:
[standard config](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/configs/dinov2/coco/panoptic/eomt_small_640.yaml),
[2x config](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/configs/dinov2/coco/panoptic/eomt_small_640_2x.yaml).
Thus, 46.7 is the **official released-model benchmark**, not a number we should
attribute to the original paper's standard-schedule row. DINOv3 and larger
backbones are also different comparisons.

## Recorded results and absolute difference

All values below are percentage-style metric points (raw JSON uses 0–1).

| Metric | Completed local evaluation | Published exact 2x reference |
|---|---:|---:|
| Images | 5,000 | COCO validation benchmark |
| PQ | 46.7364 | 46.7 |
| SQ | 81.5330 | Not supplied in the exact zoo row |
| RQ | 56.3355 | Not supplied in the exact zoo row |
| Things PQ | 50.1536 | Not supplied in the exact zoo row |
| Stuff PQ | 41.5784 | Not supplied in the exact zoo row |

Using the user-facing precision:

```text
Signed difference = 46.7364 - 46.7 = +0.0364 PQ points
Absolute difference = |46.7364 - 46.7| = 0.0364 PQ points
```

The saved score is 46.7364179389998, so the unrounded arithmetic difference
from the displayed reference is 0.0364179389998 points. This is not a measured
improvement over an unrounded author score: 46.7364 rounds to 46.7 and lies
within the ordinary rounding interval [46.65, 46.75). Comparing with the paper's
44.7 gives +2.0364 points, but that comparison mixes schedules and does not
establish an improvement over the authors' training result.

Published evidence: [verified aggregate metrics](../publication/results/verified_metrics.csv),
[source fingerprints](../publication/results/provenance.json), and the
[full evaluation record](EVALUATION.md#full-coco-val2017-evaluation-5000-images).
Original `metrics.json`, `rescored_metrics.json` and `verification.json` remain
in the local experiment archive and are not distributed.

## Comparability audit

Source inspected: upstream commit `aa017c34afbe023de9462b17ae6067b52fef9867`.
Read-only Git inspection confirmed HEAD and an empty upstream working-tree
status during this comparison.

| Component | Evidence and conclusion |
|---|---|
| Checkpoint | The zoo links `tue-mps/coco_panoptic_eomt_small_640_2x/pytorch_model.bin`. The acquisition record pins Hub revision `10f5326794a8ce7e0e975791033ae1c13aec3767`, 95,968,930 bytes, and SHA-256 `7365b41a8a129afc74c12e5dd7af534146286d02822e4b89b7bbe8ea613ff74d`. The saved full-run hash matches. This is the original upstream state dictionary, rather than the Transformers conversion. |
| Architecture and schedule identity | Config selects `vit_small_patch14_reg4_dinov2`, 200 queries and 3 joint query/image blocks; 133 output categories plus no-object. The ViT constructor uses patch size 16 despite the pretrained backbone's `patch14` name. The 24-epoch 2x label describes checkpoint training, not extra inference or test-time augmentation. |
| Input resolution | Both use a 640 x 640 model canvas. The shared upstream helper scales by `min(640/H, 640/W)`, rounds dimensions, resizes bilinearly with PIL, and zero-pads right/bottom. Images are not stretched to square. |
| Preprocessing | Local wrapper converts to RGB uint8 and calls the upstream resize/pad helper. Upstream forward divides by 255; EoMT applies ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. No second normalization or validation augmentation is applied. |
| Postprocessing | Uses upstream final-layer mask/class outputs, bilinear mask upsampling to the canvas, removal of padding, and bilinear restoration to original size. The same upstream panoptic converter uses class-confidence threshold 0.8, mask probability cutoff 0.5, overlap threshold 0.8, score-weighted mask assignment, and merging of same-category stuff. Masked attention is disabled, matching upstream evaluation instructions. |
| Category mapping | Local exporter inverts upstream `CLASS_MAPPING`: contiguous 0–79 things and 80–132 stuff become official COCO category IDs. It validates the category set and thing flags. Internal void class 133 / segment -1 becomes PNG segment ID 0; valid segment IDs are positive RGB-encoded IDs. |
| Dataset and coverage | Saved run uses COCO 2017 validation JPEGs, `panoptic_val2017.json` and native-resolution panoptic masks. Existing verification records equal image/annotation/prediction ID sets, 5,000 unique images and PNGs, valid dimensions/segments/categories, and matching input/GT hashes. All 133 categories are active: 80 things, 53 stuff. Extracted files replace upstream ZIP access without changing the intended split. |
| PQ protocol | Local evaluation uses the vendored unmodified COCO reference scorer at `7bb4655548f98f3fedc07bf37e9040a992b054b0`: same-category IoU > 0.5 matching, crowd exclusion, void correction, and dataset-level TP/FP/FN accumulation followed by category macro averaging. The upstream logged metric path is not identical; see below. |

Sources for this audit:
[checkpoint acquisition record](INFERENCE.md#official-checkpoint-acquisition-and-verification),
[official model card](https://huggingface.co/tue-mps/coco_panoptic_eomt_small_640_2x),
[local wrapper](../scripts/infer_panoptic.py),
[local evaluation/export code](../scripts/evaluate_panoptic.py),
[upstream ViT](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/models/vit.py),
[upstream EoMT](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/models/eomt.py),
[upstream panoptic evaluation step](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/training/mask_classification_panoptic.py),
[upstream transforms/converter/metrics](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/training/lightning_module.py),
[upstream COCO loader/mapping](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/datasets/coco_panoptic.py),
[official evaluation instructions](https://github.com/tue-mps/eomt/blob/aa017c34afbe023de9462b17ae6067b52fef9867/README.md#evaluating),
[COCO reference scorer](https://github.com/cocodataset/panopticapi/blob/7bb4655548f98f3fedc07bf37e9040a992b054b0/panopticapi/evaluation.py).

## Discrepancies and limits of the evidence

1. **Reporting precision explains the apparent 0.0364-point gap.** There is no
   disagreement at one decimal. The zoo does not expose its unrounded score,
   predictions, or exact scoring-run provenance for this row.
2. **Schedule explains why 44.7 is the wrong exact-match target.** The paper's
   standard small result and the zoo's longer-schedule small checkpoint are
   distinct references. This evidence supports a configuration distinction;
   it does not isolate the causal contribution of training duration locally.
3. **Scorer implementations differ.** Upstream `init_metrics_panoptic` includes
   class 133 as an extra stuff category and removes its result before reporting.
   Its custom update also contains a rule suppressing a missed target if
   `metric.void_color` covers more than half its area; the COCO scorer counts
   missed non-crowd targets as false negatives. Importantly, upstream's internal
   predicted class 133 is registered as a category, so one cannot assume that
   every prediction labeled 133 activates that void-suppression branch. These
   code differences preclude asserting identical scoring semantics. Their
   numerical effect on this run is **unmeasured**, and they are not an established
   explanation for the small displayed difference. No alternate scoring was run.
4. **Runtime parity is incomplete.** This run used CPU float32, Python 3.11.12
   and PyTorch 2.7.0+cpu. The zoo gives H100 throughput context, but does not
   fully specify the accuracy run's precision/environment. Numerical runtime
   effects are possible, not demonstrated causes. No throughput reproduction
   claim follows from this accuracy comparison.
5. **Independent rescoring is limited to scoring integrity.** Existing records
   show a separate host process directly rescored saved predictions using the
   same reference implementation, agreeing within 1e-12 for aggregate metrics.
   This is independent of model inference and the wrapper, but is not a second
   independent implementation of PQ or verification of the authors' predictions.
6. **Paper access/version limitation.** The CVF proceedings PDF endpoint returned
   HTTP 403 during this review. The paper was read through the authors' public
   arXiv v1 HTML/PDF, dated 2025-03-24, including its supplementary material.
   Differences from the final proceedings text were not verified. The exact 2x
   benchmark is established separately by the official zoo and pinned local copy.

## What was reproduced

This project has reproduced **evaluation of the authors' pretrained segmentation
checkpoint** on the full COCO 2017 validation split, with a separately verified
official-COCO score consistent with the released-model benchmark at its stated
precision. Architecture, preprocessing, postprocessing and category mapping are
comparable by source inspection; exact parity with the authors' scoring run is
not established.

This is **not an independent training reproduction**. No EoMT training or
fine-tuning was performed here. Reproducing the training result would require
training the segmentation model from the specified backbone initialization and
recipe, then evaluating the independently obtained weights. The current result
does not validate training stability, seeds, optimization, or the 2x training
schedule itself. No such additional work was undertaken for this comparison.
