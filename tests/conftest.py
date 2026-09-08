"""
Shared pytest fixtures / config for the ILSE test suite.

Tests are grouped so they can run anywhere:
- Core tests (graph math, encoders, aggregators, tuning, training on synthetic
  data) need torch, and the GNN ones skip if torch-geometric is missing.
- LLM integration tests are marked `llm` and download a small HuggingFace model
  (default EleutherAI/pythia-14m, override with $ILSE_TEST_MODEL). They skip
  gracefully if the model can't be loaded (e.g. no network on a compute node).
"""
import numpy as np
import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llm: requires downloading a HuggingFace model (needs network/cache)"
    )


@pytest.fixture(scope="session")
def device():
    """cuda when available (exercises the .to(device) flows), else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_synthetic(num_layers=7, dim=16, n_per_class=16, classes=3, seed=0):
    """Separable-ish per-layer embeddings: list of (num_layers, dim) arrays + labels."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(classes):
        center = rng.normal(size=(num_layers, dim)) * 3.0
        for _ in range(n_per_class):
            xs.append((center + rng.normal(size=(num_layers, dim)) * 0.3).astype("float32"))
            ys.append(c)
    return xs, ys


@pytest.fixture
def synthetic():
    return make_synthetic()
