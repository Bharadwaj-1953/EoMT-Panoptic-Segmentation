"""In-memory, inference-only interventions on a fully loaded EoMT-S checkpoint.

No constructors, upstream class definitions, or checkpoint files are patched.
"""
from dataclasses import dataclass
from types import MethodType

import numpy as np


@dataclass(frozen=True)
class Intervention:
    queries: int = 200
    query_blocks: int = 3
    attention: str = "off"
    query_seed: int = 0

    def validate(self):
        if self.queries not in (100, 200) or self.query_blocks not in (2, 3):
            raise ValueError("Supported queries are 100/200 and query blocks are 2/3")
        if self.attention not in ("off", "zero", "full"):
            raise ValueError("Attention must be off, zero, or full")
        if self.query_seed < 0:
            raise ValueError("Query seed must be nonnegative")
        if sum((self.queries != 200, self.query_blocks != 3, self.attention != "off")) > 1:
            raise ValueError("Change only one factor per experiment")


def select_ids(image_ids, count=500, seed=0):
    """The same uniform sample for every arm, independent of JSON ordering."""
    ordered = sorted(image_ids)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("Image IDs must be unique and nonempty")
    if count <= 0 or count > len(ordered) or seed < 0:
        raise ValueError("Sample size must be positive and <= dataset size; seed must be nonnegative")
    return sorted(int(i) for i in np.random.default_rng(seed).choice(ordered, count, replace=False))


def _deterministic_disable_mask(self, mask, probability):
    """Exact endpoint semantics: no random draws or rare rand==0 exceptions."""
    value = float(probability)
    if self.training or value not in (0.0, 1.0):
        raise RuntimeError("Endpoint attention wrapper supports eval mode and probabilities 0/1 only")
    if value == 0:
        mask[:, :self.num_q, self.num_q + self.encoder.backbone.num_prefix_tokens:] = True
    return mask


def apply_intervention(model, settings):
    """Apply once after strict original loading; return JSON-serializable provenance."""
    import torch

    settings.validate()
    network = model.network
    if getattr(network, "_sensitivity_applied", False):
        raise ValueError("Load a fresh model for every experiment")
    if model.training or network.training:
        raise ValueError("Interventions require eval mode; training is unsupported")
    if (network.num_q != 200 or network.q.num_embeddings != 200
            or network.q.weight.shape[0] != 200 or network.num_blocks != 3
            or network.masked_attn_enabled or network.attn_mask_probs.numel() != 3):
        raise ValueError("Expected a fully loaded, unmasked 200-query/3-block baseline")
    depth = len(network.encoder.backbone.blocks)
    if depth < 3:
        raise ValueError("Backbone has fewer than three blocks")
    retained = list(range(200))
    if settings.queries == 100:
        retained = select_ids(retained, 100, settings.query_seed)
        weights = network.q.weight.detach()[retained].clone()
        network.q = torch.nn.Embedding.from_pretrained(weights, freeze=True)
        network.num_q = 100
    network.num_blocks = settings.query_blocks
    if settings.attention != "off":
        network.masked_attn_enabled = True
        with torch.no_grad():
            network.attn_mask_probs.fill_(0 if settings.attention == "zero" else 1)
        # Override only this instance's mask-release helper, retaining upstream
        # mask prediction, mask construction, attention and final postprocessing.
        network._disable_attn_mask = MethodType(_deterministic_disable_mask, network)
    network._sensitivity_applied = True
    return {
        "study_type": "fixed-checkpoint inference sensitivity; not independent training",
        "queries": network.num_q, "retained_query_ids": retained,
        "query_selection_seed": settings.query_seed,
        "query_blocks": network.num_blocks, "backbone_blocks": depth,
        "query_block_indices_zero_based": list(range(depth - network.num_blocks, depth)),
        "attention": settings.attention, "masked_branch": network.masked_attn_enabled,
        "mask_probabilities": network.attn_mask_probs.tolist() if network.masked_attn_enabled else None,
        "mask_release_implementation": "deterministic endpoint wrapper" if network.masked_attn_enabled else "upstream unchanged",
        "prediction_stage": "final only", "checkpoint_modified": False,
    }
