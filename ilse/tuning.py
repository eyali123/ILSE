"""
Hyperparameter utilities for ILSE encoders.

Two public functions:

- `recommended_config(encoder_type, model_size_hint=None)`
    Zero-search defaults grounded in the ILSE paper's Optuna results. Use this
    as a starting point when you don't want to tune.

- `suggest_config(trial, encoder_type)`
    Optuna search space helper. Call from inside an Optuna `objective` to
    sample a config for the current trial. Encodes the paper's search space
    so callers don't have to re-derive it.

The user owns the training loop and the Optuna study. This module only
provides the configs.
"""
from typing import Literal, Optional

from .configs import CayleyConfig, FCConfig, SetEncoderConfig

EncoderType = Literal["cayley", "fc", "set_encoder"]
ModelSizeHint = Literal["small", "large"]


# --------------------------------------------------------------------------- #
# Zero-search defaults
# --------------------------------------------------------------------------- #

def recommended_config(
    encoder_type: EncoderType,
    model_size_hint: Optional[ModelSizeHint] = None,
):
    """
    Return an empirically-validated default config from the ILSE paper.

    Args:
        encoder_type: One of "cayley", "fc", "set_encoder".
        model_size_hint: Optional hint about backbone size.
            - "small": backbones <~1B params (e.g., Pythia-410m, TinyLlama-1.1B).
              Uses lr=1e-3.
            - "large": backbones >=~3B params (e.g., Pythia-2.8b, Llama3-8B).
              Uses lr=1e-4 (larger models benefit from smaller learning rates
              in our search).
            - None: defaults to "small" (lr=1e-3).

    Returns:
        A CayleyConfig / FCConfig / SetEncoderConfig with fields populated.

    Notes:
        These defaults come from sequence-level classification experiments.
        For other task types (per-token regression, ordinal, retrieval), they
        are a reasonable starting point but may not be optimal -- run
        `suggest_config` under Optuna for those regimes if budget allows.
    """
    lr = 1e-4 if model_size_hint == "large" else 1e-3

    if encoder_type == "cayley":
        return CayleyConfig(
            conv_type="gin",
            hidden_dim=256,
            gnn_layers=1,
            gin_mlp_layers=1,
            gat_heads=4,
            pooling="mean",
            dropout=0.1,
            lr=lr,
            weight_decay=1e-4,
        )

    if encoder_type == "fc":
        return FCConfig(
            conv_type="gin",
            hidden_dim=256,
            gnn_layers=1,
            gin_mlp_layers=1,
            gat_heads=4,
            pooling="mean",
            dropout=0.1,
            lr=lr,
            weight_decay=1e-4,
        )

    if encoder_type == "set_encoder":
        return SetEncoderConfig(
            hidden_dim=256,
            pre_pooling_layers=1,
            post_pooling_layers=1,
            pooling="sum",
            dropout=0.2,
            lr=lr,
            weight_decay=1e-4,
        )

    raise ValueError(
        f"Unknown encoder_type={encoder_type!r}. "
        f"Expected one of: 'cayley', 'fc', 'set_encoder'."
    )


# --------------------------------------------------------------------------- #
# Optuna search space
# --------------------------------------------------------------------------- #

def suggest_config(trial, encoder_type: EncoderType):
    """
    Sample one ILSE config from the paper's search space for an Optuna trial.

    Usage:
        import optuna
        from ilse.tuning import suggest_config

        def objective(trial):
            cfg = suggest_config(trial, "cayley")
            model = build_my_model_with_aggregator(cfg)
            return train_and_eval(model)  # user's code

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)

    Args:
        trial: An Optuna `Trial` object.
        encoder_type: "cayley", "fc", or "set_encoder".

    Returns:
        CayleyConfig / FCConfig / SetEncoderConfig with values picked for
        this trial.

    Notes:
        - The search space covers encoder-structural params (hidden_dim,
          gnn_layers, pooling, dropout) plus optimizer params (lr,
          weight_decay). It does NOT search batch_size or epochs -- those
          are usually fixed by compute budget and dataset size.
        - For Cayley/FC, `conv_type` is searched over ("gin", "gcn").
          `gin_mlp_layers` is only searched when `conv_type == "gin"`.
        - Ranges match the ILSE paper's sweeps. If you want to widen or
          narrow the space, write your own suggest function -- this one is
          intentionally simple and not parameterized.
    """
    if encoder_type in ("cayley", "fc"):
        cfg_cls = CayleyConfig if encoder_type == "cayley" else FCConfig

        conv_type = trial.suggest_categorical("conv_type", ["gin", "gcn", "gat"])
        gin_mlp_layers = (
            trial.suggest_int("gin_mlp_layers", 1, 2) if conv_type == "gin" else 0
        )
        gat_heads = (
            trial.suggest_categorical("gat_heads", [2, 4, 8]) if conv_type == "gat" else 4
        )
        # GAT empirically benefits from higher weight_decay (1e-3 dominant in search)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        return cfg_cls(
            conv_type=conv_type,
            hidden_dim=trial.suggest_categorical("hidden_dim", [128, 256, 512]),
            gnn_layers=trial.suggest_int("gnn_layers", 1, 3),
            gin_mlp_layers=gin_mlp_layers,
            gat_heads=gat_heads,
            pooling=trial.suggest_categorical("pooling", ["mean", "sum", "last"]),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            weight_decay=weight_decay,
        )

    if encoder_type == "set_encoder":
        return SetEncoderConfig(
            hidden_dim=trial.suggest_categorical("hidden_dim", [128, 256, 512]),
            pre_pooling_layers=trial.suggest_int("pre_pooling_layers", 0, 2),
            post_pooling_layers=trial.suggest_int("post_pooling_layers", 0, 2),
            pooling=trial.suggest_categorical("pooling", ["mean", "sum"]),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        )

    raise ValueError(
        f"Unknown encoder_type={encoder_type!r}. "
        f"Expected one of: 'cayley', 'fc', 'set_encoder'."
    )
