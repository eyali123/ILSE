"""
Small helpers to infer aggregator sizing from a HuggingFace model.

`build_aggregator` / `build_per_token_aggregator` need `num_layers` and
`hidden_in`. Rather than hardcoding them (and getting the +1 embedding-layer
offset wrong), pass a loaded model or a model name/path to these helpers.
"""
from typing import Union


def _resolve_config(model_or_name, trust_remote_code: bool = False):
    """Return a HF config from a loaded model, a config object, or a name/path."""
    # A loaded model (has .config) or a config object (has num_hidden_layers).
    cfg = getattr(model_or_name, "config", model_or_name)
    if hasattr(cfg, "num_hidden_layers") or hasattr(cfg, "hidden_size"):
        return cfg
    if isinstance(model_or_name, str):
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(model_or_name, trust_remote_code=trust_remote_code)
    raise TypeError(
        "Expected a HuggingFace model, a config, or a model name/path; "
        f"got {type(model_or_name).__name__}"
    )


def num_layers_for(model_or_name: Union[str, object], trust_remote_code: bool = False) -> int:
    """
    Number of layer-nodes ILSE aggregates for a model: hidden layers + 1.

    The +1 is the embedding-layer output, which HuggingFace includes in
    `output_hidden_states`. This is exactly the `num_layers` argument for
    `build_aggregator` / `build_per_token_aggregator`.

    Args:
        model_or_name: A loaded HF model, a HF config, or a model name/path.
        trust_remote_code: Forwarded to AutoConfig when a name/path is given.

    Example:
        >>> from ilse import build_aggregator, CayleyConfig, num_layers_for, hidden_size_for
        >>> name = "EleutherAI/pythia-410m"
        >>> agg = build_aggregator(CayleyConfig(),
        ...                        num_layers=num_layers_for(name),
        ...                        hidden_in=hidden_size_for(name))
    """
    cfg = _resolve_config(model_or_name, trust_remote_code)
    return cfg.num_hidden_layers + 1


def hidden_size_for(model_or_name: Union[str, object], trust_remote_code: bool = False) -> int:
    """
    Backbone hidden size for a model -- the `hidden_in` argument for the
    aggregator factories.

    Args:
        model_or_name: A loaded HF model, a HF config, or a model name/path.
        trust_remote_code: Forwarded to AutoConfig when a name/path is given.
    """
    cfg = _resolve_config(model_or_name, trust_remote_code)
    return cfg.hidden_size
