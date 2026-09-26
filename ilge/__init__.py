"""
ILGE: Inter-Layer Geometry Encoders

Lightweight encoders (~300K-600K params) that aggregate frozen LLM layer
embeddings. No fine-tuning of the base model required.

Three encoder families:
- CayleyConfig:     GNN with algebraic Cayley graph topology (SL(2, Z_n))
- FCConfig:         GNN with fully-connected graph topology
- DeepSetConfig:    DeepSet (permutation-invariant, no message passing)

Three entry points:

1) Sklearn-style classifier for simple sequence classification:

    from ilge import ILGEClassifier, CayleyConfig
    clf = ILGEClassifier("EleutherAI/pythia-410m", CayleyConfig(), num_classes=6)
    clf.fit(train_texts, train_labels)

2) Pure nn.Module for custom fine-tuning pipelines (per-token, regression,
   ordinal, retrieval, etc.) -- you own the backbone and training loop:

    from ilge import build_aggregator, CayleyConfig
    agg = build_aggregator(CayleyConfig(), num_layers=37, hidden_in=2560)
    # agg: (N, num_layers, hidden_in) -> (N, agg.out_dim)

3) Per-token aggregator with cross-token Cayley graph (e.g., per-residue
   protein tasks where you want GNN mixing across all tokens and layers):

    from ilge import build_per_token_aggregator, CayleyConfig
    agg = build_per_token_aggregator(CayleyConfig(conv_type="gat"), num_layers=37, hidden_in=2560)
    # agg: (batch, seq_len, num_layers, hidden_in) -> (batch, seq_len, agg.out_dim)

Hyperparameter helpers (optional):

    from ilge.tuning import recommended_config, suggest_config
    cfg = recommended_config("cayley", model_size_hint="large")   # paper defaults
    # or in an Optuna objective:
    cfg = suggest_config(trial, "cayley")                          # search space
"""

__version__ = "0.1.0"

from .configs import CayleyConfig, FCConfig, DeepSetConfig
from .classifier import ILGEClassifier
from .aggregator import (
    Aggregator,
    PerTokenAggregator,
    build_aggregator,
    build_per_token_aggregator,
)
from .model_utils import num_layers_for, hidden_size_for
from . import tuning

__all__ = [
    "ILGEClassifier",
    "CayleyConfig",
    "FCConfig",
    "DeepSetConfig",
    "Aggregator",
    "PerTokenAggregator",
    "build_aggregator",
    "build_per_token_aggregator",
    "num_layers_for",
    "hidden_size_for",
    "tuning",
]
