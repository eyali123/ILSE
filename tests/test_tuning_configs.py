"""recommended_config / suggest_config: types, lr-by-size, no 'last' pooling."""
import pytest

from ilse import CayleyConfig, FCConfig, SetEncoderConfig
from ilse.tuning import recommended_config, suggest_config


def test_recommended_types_and_lr():
    assert isinstance(recommended_config("cayley"), CayleyConfig)
    assert recommended_config("cayley").lr == 1e-3
    assert recommended_config("cayley", model_size_hint="large").lr == 1e-4
    assert isinstance(recommended_config("fc"), FCConfig)
    assert isinstance(recommended_config("set_encoder"), SetEncoderConfig)


def test_recommended_bad_type():
    with pytest.raises(ValueError):
        recommended_config("nope")


class FakeTrial:
    """Minimal Optuna Trial stand-in: returns the first/low value each time."""

    def __init__(self, conv_type="gin"):
        self._conv = conv_type

    def suggest_categorical(self, name, choices):
        if name == "conv_type":
            return self._conv
        return choices[0]

    def suggest_int(self, name, low, high):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low


@pytest.mark.parametrize(
    "etype,conv",
    [("cayley", "gin"), ("cayley", "gcn"), ("cayley", "gat"), ("fc", "gin")],
)
def test_suggest_gnn_no_last_pooling(etype, conv):
    cfg = suggest_config(FakeTrial(conv), etype)
    assert cfg.conv_type == conv
    assert cfg.pooling in ("mean", "sum")   # "last" must not be sampled


def test_suggest_set_encoder():
    cfg = suggest_config(FakeTrial(), "set_encoder")
    assert isinstance(cfg, SetEncoderConfig)
    assert cfg.pooling in ("mean", "sum")


def test_suggest_bad_type():
    with pytest.raises(ValueError):
        suggest_config(FakeTrial(), "nope")
