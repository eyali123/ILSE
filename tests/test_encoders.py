"""Encoder modules: SetEncoder, GNNEncoder construction, real-node pooling."""
import types

import pytest
import torch

from ilse import CayleyConfig
from ilse._internal.nn_modules import SetEncoder, build_gnn_encoder


# --------------------------------------------------------------------------- #
# SetEncoder (no PyG needed)
# --------------------------------------------------------------------------- #

def test_setencoder_shapes():
    enc = SetEncoder(num_layers=7, layer_dim=16, hidden_dim=32).eval()
    y = enc(torch.randn(4, 7, 16))
    assert y.shape == (4, 32)
    assert enc.out_dim == 32


def test_setencoder_no_mlps_returns_pooled_dim():
    enc = SetEncoder(7, 16, hidden_dim=32, pre_pooling_layers=0, post_pooling_layers=0).eval()
    y = enc(torch.randn(2, 7, 16))
    assert y.shape == (2, 16)
    assert enc.out_dim == 16


def test_setencoder_rejects_bad_pooling():
    with pytest.raises(ValueError):
        SetEncoder(7, 16, pooling="last")


# --------------------------------------------------------------------------- #
# GNNEncoder construction / validation
# --------------------------------------------------------------------------- #

def test_gnn_rejects_last_pooling_without_pyg():
    # pooling is validated before torch-geometric is imported, so this runs
    # even where PyG is absent. Regression test for "last" removal.
    with pytest.raises(ValueError):
        build_gnn_encoder(CayleyConfig(pooling="last"), in_dim=16)


def test_gat_heads_propagate():
    # Regression test: gat_heads must reach the GATConv (was dropped on the
    # classifier path before the shared build_gnn_encoder helper).
    pytest.importorskip("torch_geometric")
    enc = build_gnn_encoder(CayleyConfig(conv_type="gat", gat_heads=8, hidden_dim=16), in_dim=16)
    assert enc.convs[0].heads == 8


def test_pooling_excludes_virtual_nodes(device):
    # With gnn_layers=0 the forward is just proj_in -> act -> pool, so we can
    # check exactly that virtual (is_real=False) nodes are excluded from pooling.
    pytest.importorskip("torch_geometric")
    torch.manual_seed(0)
    enc = build_gnn_encoder(
        CayleyConfig(conv_type="gin", gnn_layers=0, hidden_dim=8, pooling="sum"),
        in_dim=8,
    ).to(device).eval()

    x = torch.randn(6, 8, device=device)  # 2 graphs x (2 real + 1 virtual)
    batch = types.SimpleNamespace(
        x=x,
        edge_index=torch.zeros(2, 0, dtype=torch.long, device=device),  # unused (0 conv layers)
        batch=torch.tensor([0, 0, 0, 1, 1, 1], device=device),
        num_graphs=2,
        is_real=torch.tensor([True, True, False, True, True, False], device=device),
    )
    out = enc(batch)

    h = enc.act(enc.proj_in(x))  # dropout is identity in eval
    expected0 = h[[0, 1]].sum(dim=0)
    expected1 = h[[3, 4]].sum(dim=0)
    assert torch.allclose(out[0], expected0, atol=1e-5)
    assert torch.allclose(out[1], expected1, atol=1e-5)
