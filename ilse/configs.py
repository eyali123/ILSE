"""
Configuration dataclasses for ILSE encoders.

Defaults are set from Optuna hyperparameter search results across
3 LLMs (Pythia-410m, TinyLlama-1.1B, Llama3-8B) and 6 classification tasks.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class CayleyConfig:
    """
    GNN encoder with Cayley graph topology (SL(2, Z_n) algebraic graph).

    Sparse, regular connectivity. Adapts automatically to any number of layers
    by finding the smallest Cayley graph >= num_layers and padding with virtual nodes.

    Defaults grounded in Optuna search across 18 (model, task) combos:
    - gnn_layers=1 wins in ~14/18 cases
    - pooling="mean" wins in 14/18 cases (GIN and GCN)
    - lr=1e-3 is best for models up to ~1B; try 1e-4 for larger models (>3B)
    """

    conv_type: Literal["gin", "gcn"] = "gin"
    hidden_dim: int = 256
    gnn_layers: int = 1
    gin_mlp_layers: int = 1  # only used when conv_type="gin"; ignored for "gcn"
    pooling: Literal["mean", "sum", "last"] = "mean"
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 50


@dataclass
class FCConfig:
    """
    GNN encoder with Fully-Connected graph topology (all layers see each other).

    Denser than Cayley; slightly more expensive for many layers.

    Defaults grounded in Optuna search across 18 (model, task) combos:
    - gnn_layers=1 is dominant
    - pooling="mean" is a safe default (data shows a split between "mean" and "last")
    - If performance is below expectation, try pooling="last"
    """

    conv_type: Literal["gin", "gcn"] = "gin"
    hidden_dim: int = 256
    gnn_layers: int = 1
    gin_mlp_layers: int = 1  # only used when conv_type="gin"; ignored for "gcn"
    pooling: Literal["mean", "sum", "last"] = "mean"
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 50


@dataclass
class SetEncoderConfig:
    """
    DeepSet encoder: phi(per-layer MLP) -> pool -> rho(post-pool MLP).

    Permutation-invariant (no message passing between layers).
    Stronger than simple baselines; competitive with GNN encoders on many tasks.

    Defaults grounded in Optuna search across 18 (model, task) combos:
    - pre_pooling_layers=1 in 17/18 cases
    - pooling="sum" dominant for classification; use "mean" for STS/regression
    - dropout=0.2 (slightly higher than GNN variants)
    """

    hidden_dim: int = 256
    pre_pooling_layers: int = 1
    post_pooling_layers: int = 1
    pooling: Literal["mean", "sum"] = "sum"
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 50
