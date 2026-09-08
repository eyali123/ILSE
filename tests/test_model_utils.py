"""num_layers_for / hidden_size_for: model, config, and error paths."""
import pytest

from ilse import num_layers_for, hidden_size_for


class _Cfg:
    num_hidden_layers = 6
    hidden_size = 128


class _Model:
    config = _Cfg()


def test_from_loaded_model():
    assert num_layers_for(_Model()) == 7        # 6 hidden + 1 embedding
    assert hidden_size_for(_Model()) == 128


def test_from_config_object():
    assert num_layers_for(_Cfg()) == 7
    assert hidden_size_for(_Cfg()) == 128


def test_bad_input():
    with pytest.raises(TypeError):
        num_layers_for(123)
