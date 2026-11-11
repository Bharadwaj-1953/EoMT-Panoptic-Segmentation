"""Offline CPU evaluation of fixed-checkpoint sensitivities, one arm per new run."""
import argparse
from dataclasses import asdict
import json
import sys
import time

if __package__:
    from . import evaluate_panoptic as evaluation
    from .ablation_interventions import Intervention, apply_intervention, select_ids
else:
    import evaluate_panoptic as evaluation
    from ablation_interventions import Intervention, apply_intervention, select_ids


def evaluate(args, *, loader=None, predictor=None, mapping=None):
    settings = Intervention(args.queries, args.query_blocks, args.attention, args.query_seed)
    settings.validate()
    if args.sample_size <= 0 or args.sample_seed < 0 or args.threads <= 0:
        raise ValueError("Sample size and threads must be positive; sample seed must be nonnegative")
    import torch

    torch.set_num_threads(args.threads)
    # No inter-op override: it cannot reliably be changed after parallel work.
    # Record its actual value; launch a fresh process for each arm.
    experiment = {
        "requested": asdict(settings),
        "selection": {"method": "numpy.default_rng.choice without replacement on sorted IDs; sorted output",
                      "seed": args.sample_seed, "count": args.sample_size, "numpy": evaluation.np.__version__},
        "command": list(sys.argv), "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "source_hashes": {str(path.relative_to(evaluation.inference.ROOT)): evaluation.inference.file_sha256(path)
                          for path in (evaluation.Path(__file__),
                                       evaluation.Path(__file__).with_name("ablation_interventions.py"),
                                       evaluation.Path(evaluation.__file__),
                                       evaluation.Path(evaluation.inference.__file__))},
    }

    def load_and_intervene(checkpoint):
        model = (loader or evaluation.inference.load_model)(checkpoint)
        tick = time.perf_counter()
        experiment["effective"] = apply_intervention(model, settings)
        experiment["intervention_seconds"] = time.perf_counter() - tick
        return model

    return evaluation.evaluate(
        args, loader=load_and_intervene, predictor=predictor, mapping=mapping,
        selector=lambda ids: select_ids(ids, args.sample_size, args.sample_seed),
        experiment=experiment, output_root="results/ablations")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="data/coco/val2017")
    parser.add_argument("--panoptic", default="data/coco/annotations/panoptic_val2017")
    parser.add_argument("--annotations", default="data/coco/annotations/panoptic_val2017.json")
    parser.add_argument("--checkpoint", default=evaluation.inference.DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", required=True, help="New directory under results/ablations/")
    parser.add_argument("--queries", type=int, choices=(100, 200), default=200)
    parser.add_argument("--query-blocks", type=int, choices=(2, 3), default=3)
    parser.add_argument("--attention", choices=("off", "zero", "full"), default="off")
    parser.add_argument("--query-seed", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    # Compatibility with the shared evaluator; its selector overrides first-N.
    args.limit, args.full = args.sample_size, False
    try:
        result = evaluate(args)
        print(json.dumps(result["metrics"], indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        print(f"Ablation evaluation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
