"""
End-to-end ILSEClassifier + LLMLayerExtractor against a small real LLM.

Marked `llm`: downloads $ILSE_TEST_MODEL (default EleutherAI/pythia-14m).
Skips gracefully if the model cannot be loaded (e.g. no network on the node).
Run just these with:  pytest -m llm
Skip them with:       pytest -m "not llm"
"""
import os

import numpy as np
import pytest
import torch

from ilse import ILSEClassifier, CayleyConfig, SetEncoderConfig

MODEL = os.environ.get("ILSE_TEST_MODEL", "EleutherAI/pythia-14m")


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _toy_dataset():
    texts = (
        ["i absolutely love this, wonderful"] * 10
        + ["this is terrible, awful and bad"] * 10
        + ["the meeting is scheduled at noon"] * 10
    )
    labels = [0] * 10 + [1] * 10 + [2] * 10
    return texts, labels


@pytest.mark.llm
@pytest.mark.parametrize(
    "config",
    [
        CayleyConfig(hidden_dim=32, gnn_layers=1, epochs=5, batch_size=8),
        SetEncoderConfig(hidden_dim=32, epochs=5, batch_size=8),
    ],
    ids=["cayley", "set_encoder"],
)
def test_classifier_end_to_end(tmp_path, config):
    if isinstance(config, CayleyConfig):
        pytest.importorskip("torch_geometric")
    pytest.importorskip("transformers")

    clf = ILSEClassifier(MODEL, config, num_classes=3, device=_dev())
    try:
        clf._get_extractor()  # triggers tokenizer/model load (download)
    except Exception as e:  # noqa: BLE001 - network / auth / missing model
        pytest.skip(f"could not load {MODEL}: {e}")

    texts, labels = _toy_dataset()
    clf.fit(texts, labels, verbose=False)

    preds = clf.predict(texts)
    assert len(preds) == len(texts)
    assert all(0 <= p < 3 for p in preds)

    proba = clf.predict_proba(texts)
    assert proba.shape == (len(texts), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    acc = clf.score(texts, labels)
    assert 0.0 <= acc <= 1.0

    # save / load round-trip must reproduce predictions exactly.
    path = tmp_path / "model.pt"
    clf.save(str(path))
    clf2 = ILSEClassifier.load(str(path), device=_dev())
    assert clf2.predict(texts) == preds


@pytest.mark.llm
@pytest.mark.parametrize("pooling", ["mean", "last", "first", "mean_including_padding"])
def test_token_pooling_variants(pooling):
    pytest.importorskip("transformers")
    from ilse._internal.llm import LLMLayerExtractor

    try:
        ext = LLMLayerExtractor(MODEL, device=_dev(), token_pooling=pooling)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"could not load {MODEL}: {e}")

    embs = ext.extract(["hello world", "a b c d e f"], batch_size=2, show_progress=False)
    assert len(embs) == 2
    for e in embs:
        assert e.shape == (ext.num_layers, ext.hidden_dim)
    ext.unload()


def test_bad_token_pooling_rejected():
    pytest.importorskip("transformers")
    from ilse._internal.llm import LLMLayerExtractor

    # Validated before any model load, so this needs no network.
    with pytest.raises(ValueError):
        LLMLayerExtractor(MODEL, token_pooling="bogus")
