# Verified full-validation masking results

The [published comparison](../publication/results/masking_comparison.csv) and [aggregate metrics](../publication/results/verified_metrics.csv) retain the verified numerical results. Paths under `results/` below identify local evidence; masks, logs and detailed audit files are not distributed.

Verified on 2026-11-11 UTC. **Full M1 achieves 45.6785 PQ on all 5,000 COCO
validation images, a decrease of 1.0579 points against verified full B0.**
The adverse direction agrees with the matched 500-image pilot, whose loss was
1.3502 points. These are fixed-checkpoint inference interventions.

## Execution evidence

The existing output is `results/ablations/coco-val5000-mask-full`. Its manifest
started at 2026-11-11T01:38:47.866560+00:00 and requests 5,000 images, 200 queries,
three joint blocks and `attention=full`. Effective settings specify masked
attention with probabilities [1,1,1], joint blocks [9,10,11] zero-based and
final-stage predictions. The checkpoint fingerprint and current recorded
source hashes match the intended implementation.

The audit continued read-only filesystem monitoring at approximately one-minute
intervals until 05:26:27 UTC, when both metadata records were complete and all
5,000 PNGs existed. The evaluator's recorded finish time is
2026-11-11T05:26:15.856662+00:00. Its original artifacts were then independently
verified and rescored in a separate host process. Verification passed at
05:31:01 UTC. Before/after hashes confirmed that its 5,007 original files were
unchanged by verification.

Docker engine-pipe access was denied during verification, so the container's
state, logs and exit code could not be inspected. No exit-code claim is made;
artifact completion and numerical integrity were established independently.
The completed progress history is in
`results/audits/final-20261111-0248/progress.jsonl`.

## Verified full-validation comparison

Both arms were freshly verified from all saved masks in separate host processes.
Each uses all 5,000 validation IDs and all 133 active categories, including
80 things and 53 stuff. Values and signed deltas below are 0–100 points.

| Full-validation metric | B0, 5,000 images | M1, 5,000 images | M1 minus B0 |
|---|---:|---:|---:|
| All PQ | 46.7364 | 45.6785 | −1.0579 |
| All SQ | 81.5330 | 81.2838 | −0.2492 |
| All RQ | 56.3355 | 54.8686 | −1.4669 |
| Things PQ | 50.1536 | 49.2699 | −0.8837 |
| Stuff PQ | 41.5784 | 40.2576 | −1.3208 |

Baseline reference evidence is
`results/audits/final-20261111-0248/baseline-full/verification.json` and
`rescored_metrics.json`. Full M1 evidence is in the sibling
`mask-full-5000/verification.json`, `rescored_metrics.json`,
`prediction_manifest.json`, `adapter.json` and `input_artifact_integrity.json`.
The comparison is exported to `masking_comparison.json` and `.csv` in the audit
directory, with raw group scores and signed point differences.

The full M1 scorer counted 29,659 TP, 8,825 FP and 26,636 FN, compared with
30,677 TP, 9,723 FP and 25,618 FN for full B0. M1 exported 41,243 segments.
The checks establish exact ID/file/timing coverage, native RGB dimensions,
segment IDs/areas/categories, source/checkpoint/config/annotation identities,
per-image input/GT hashes and CSV/JSON agreement. Direct rescoring matches
aggregate and per-class PQ/SQ/RQ within 1e-12, counts exactly and per-class
accumulated IoU within 1e-8. Supplemental CSV category and metadata checks passed.

## Verified 500-image pilot context

| Metric | Pilot B0 | Pilot M1 | M1 minus matched B0 |
|---|---:|---:|---:|
| All PQ | 47.7820 | 46.4318 | −1.3502 |
| All SQ | 81.6096 | 80.3132 | −1.2964 |
| All RQ | 57.1361 | 55.3781 | −1.7580 |
| Things PQ | 50.8259 | 50.1985 | −0.6274 |
| Stuff PQ | 43.3024 | 40.8885 | −2.4139 |

Both pilot arms were freshly checked and rescored in this audit. These values
describe identical 500-image inputs and 131 active categories. The full arms
average 133 categories. The pilot is included within the full validation set,
so the expansion is an overlapping-population confirmation.

Full M1's PQ effect is 0.2923 points less adverse than the pilot effect. Its
Things PQ loss is larger (0.8837 versus 0.6274), while its Stuff PQ loss is smaller
(1.3208 versus 2.4139). These are descriptive comparisons across the two
populations. The expanded result retains the negative overall direction and
does not establish statistical significance, a training-time annealing benefit
or the performance of independently trained always-masked models.

## Execution and verification timing

| Measured interval | Seconds |
|---|---:|
| Evaluator counter interval | 13,649.4638 |
| UTC start-to-finish interval | 13,647.9901 |
| Sum of prediction-call intervals | 12,838.2590 |
| Mean prediction-call duration | 2.5677 |
| Median prediction-call duration | 2.3708 |
| Independent verifier, including rescoring | 217.1965 |
| Independent rescoring alone | 142.3690 |

The evaluator took about 3 hours 47 minutes 29 seconds. Prediction-call timing
includes preprocessing, forward and postprocessing. The verifier timer excludes
the adapter's preflight and before/after complete-original-directory hashing.
CPU float32, four intra-op/inter-op threads and the recorded offline configuration
were retained. This one run was observed while other audit work occurred on the
host, so timing comparisons with the historical baseline are uncontrolled.

The executed adapted host verifier uses no model imports or inference and directs
outputs exclusively to
`results/audits/final-20261111-0248/mask-full-5000/` with exclusive creation.
The exact adapter command is retained in the local audit archive. It was run only after complete artifacts
were available. Original experiment outputs, checkpoint and upstream files were
preserved. Remaining operational limitations are Docker exit/log inspection and
complete test-suite revalidation in the dependency-equipped environment.
