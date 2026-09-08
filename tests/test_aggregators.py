"""Public aggregator factories: shapes, grads, per-token modes, validation."""
import pytest
import torch

from ilse import (
    build_aggregator,
    build_per_token_aggregator,
    CayleyConfig,
    FCConfig,
    SetEncoderConfig,
)

GNN_CONFIGS = [
    CayleyConfig(hidden_dim=16, gnn_layers=1),
    CayleyConfig(conv_type="gcn", hidden_dim=16),
    CayleyConfig(conv_type="gat", gat_heads=4, hidden_dim=16),
    FCConfig(hidden_dim=16),
    FCConfig(conv_type="gat", gat_heads=2, hidden_dim=16),
]


@pytest.mark.parametrize("cfg", GNN_CONFIGS, ids=lambda c: f"{type(c).__name__}-{c.conv_type}")
def test_build_aggregator_gnn_forward_and_grad(cfg, device):
    pytest.importorskip("torch_geometric")
    agg = build_aggregator(cfg, num_layers=7, hidden_in=16).to(device)
    x = torch.randn(5, 7, 16, device=device, requires_grad=True)
    y = agg(x)
    assert y.shape == (5, 16)
    assert agg.out_dim == 16
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_build_aggregator_setencoder_forward(device):
    agg = build_aggregator(SetEncoderConfig(hidden_dim=16), num_layers=7, hidden_in=16).to(device)
    y = agg(torch.randn(5, 7, 16, device=device))
    assert y.shape == (5, 16)


def test_aggregator_shape_validation(device):
    agg = build_aggregator(SetEncoderConfig(hidden_dim=16), num_layers=7, hidden_in=16).to(device)
    with pytest.raises(ValueError):
        agg(torch.randn(5, 6, 16, device=device))   # wrong num_layers
    with pytest.raises(ValueError):
        agg(torch.randn(5, 7, 8, device=device))    # wrong hidden_in


@pytest.mark.parametrize("conv", ["gin", "gcn", "gat"])
def test_per_token_modes(conv, device):
    pytest.importorskip("torch_geometric")
    cfg = CayleyConfig(conv_type=conv, gat_heads=4, hidden_dim=16, gnn_layers=1)

    per_tok = build_per_token_aggregator(
        cfg, num_layers=7, hidden_in=16, output_mode="per_token"
    ).to(device)
    x = torch.randn(2, 8, 7, 16, device=device)  # (B, T, L, D); T*L=56 -> 120-node Cayley
    y = per_tok(x)
    assert y.shape == (2, 8, 16)

    seq = build_per_token_aggregator(
        cfg, num_layers=7, hidden_in=16, output_mode="sequence"
    ).to(device)
    y2 = seq(x)
    assert y2.shape == (2, 16)


def test_per_token_rejects_non_cayley():
    with pytest.raises(TypeError):
        build_per_token_aggregator(FCConfig(), num_layers=7, hidden_in=16)


def test_per_token_rejects_bad_modes():
    with pytest.raises(ValueError):
        build_per_token_aggregator(CayleyConfig(), 7, 16, output_mode="nope")
    with pytest.raises(ValueError):
        build_per_token_aggregator(CayleyConfig(), 7, 16, token_pooling="nope")
