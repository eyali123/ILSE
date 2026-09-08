"""Training loop + datasets on synthetic data (no LLM needed)."""
import pytest
import torch
from torch.utils.data import DataLoader

from ilse import CayleyConfig
from ilse._internal.nn_modules import SetEncoder, ClassificationHead, build_gnn_encoder
from ilse._internal.dataset import (
    TensorDataset,
    GraphDataset,
    tensor_collate,
    graph_collate,
)
from ilse._internal.training import train_model

from conftest import make_synthetic


def test_train_setencoder_synthetic(device):
    xs, ys = make_synthetic()
    ds = TensorDataset(xs, ys)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=tensor_collate)

    enc = SetEncoder(num_layers=7, layer_dim=16, hidden_dim=32)
    head = ClassificationHead(enc.out_dim, num_classes=3)
    result = train_model(
        enc, head, loader, loader,
        epochs=8, device=device, is_gnn=False, verbose=False,
    )
    assert result["best_state"] is not None
    assert 0.0 <= result["best_val_acc"] <= 1.0


def test_train_gnn_synthetic(device):
    pytest.importorskip("torch_geometric")
    xs, ys = make_synthetic()
    ds = GraphDataset(xs, ys, topology="cayley")
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=graph_collate)

    enc = build_gnn_encoder(CayleyConfig(hidden_dim=32, gnn_layers=1), in_dim=16)
    head = ClassificationHead(enc.out_dim, num_classes=3)
    result = train_model(
        enc, head, loader, loader,
        epochs=8, device=device, is_gnn=True, verbose=False,
    )
    assert result["best_state"] is not None
    assert 0.0 <= result["best_val_acc"] <= 1.0


def test_graph_dataset_pads_and_marks_real():
    pytest.importorskip("torch_geometric")
    xs, ys = make_synthetic(num_layers=7, dim=16, n_per_class=1, classes=1)
    ds = GraphDataset(xs, ys, topology="cayley")
    data = ds[0]
    # 7 real layers padded up to the 24-node Cayley graph (n=3).
    assert data.x.size(0) == ds.num_graph_nodes == 24
    assert data.is_real.sum().item() == 7
    assert data.is_real[:7].all() and not data.is_real[7:].any()
