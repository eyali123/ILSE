"""
ILSE: Intermediate Layer Structure Encoders

Lightweight encoders (~300K-600K params) that aggregate frozen LLM layer
embeddings. No fine-tuning of the base model required.

Three encoder families:
- CayleyConfig:     GNN with algebraic Cayley graph topology (SL(2, Z_n))
- FCConfig:         GNN with fully-connected graph topology
- SetEncoderConfig: DeepSet (permutation-invariant, no message passing)

Two entry points:

1) Sklearn-style classifier for simple sequence classification:

    from ilse import ILSEClassifier, CayleyConfig
    clf = ILSEClassifier("EleutherAI/pythia-410m", CayleyConfig(), num_classes=6)
    clf.fit(train_texts, train_labels)

2) Pure nn.Module for custom fine-tuning pipelines (per-token, regression,
   ordinal, retrieval, etc.) -- you own the backbone and training loop:

    from ilse import build_aggregator, CayleyConfig
    agg = build_aggregator(CayleyConfig(), num_layers=37, hidden_in=2560)
    # agg: (N, num_layers, hidden_in) -> (N, agg.out_dim)

3) Per-token aggregator with cross-token Cayley graph (e.g., per-residue
   protein tasks where you want GNN mixing across all tokens and layers):

    from ilse import build_per_token_aggregator, CayleyConfig
    agg = build_per_token_aggregator(CayleyConfig(conv_type="gat"), num_layers=37, hidden_in=2560)
    # agg: (batch, seq_len, num_layers, hidden_in) -> (batch, seq_len, agg.out_dim)

Hyperparameter helpers (optional):

    from ilse.tuning import recommended_config, suggest_config
    cfg = recommended_config("cayley", model_size_hint="large")   # paper defaults
    # or in an Optuna objective:
    cfg = suggest_config(trial, "cayley")                          # search space
"""

__version__ = "0.1.0"

from .configs import CayleyConfig, FCConfig, SetEncoderConfig
from .classifier import ILSEClassifier
from .aggregator import (
    Aggregator,
    PerTokenAggregator,
    build_aggregator,
    build_per_token_aggregator,
)
from . import tuning

__all__ = [
    "ILSEClassifier",
    "CayleyConfig",
    "FCConfig",
    "SetEncoderConfig",
    "Aggregator",
    "PerTokenAggregator",
    "build_aggregator",
    "build_per_token_aggregator",
    "tuning",
]
