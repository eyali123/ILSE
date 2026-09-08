"""
Dataset classes for ILSE encoders.

- GraphDataset: wraps layer embeddings as PyG graphs (for GNN encoders)
- TensorDataset: keeps [L, D] tensors (for SetEncoder)
"""
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from .graph_ops import build_edge_index


class GraphDataset(TorchDataset):
    """
    Dataset for GNN encoders (Cayley, FC).

    Each sample is a PyG graph where nodes = LLM layers, edges = topology.
    """

    def __init__(
        self,
        layerwise_list: List[np.ndarray],
        labels: List[int],
        topology: str,
    ):
        """
        Args:
            layerwise_list: List of arrays, each shape (num_layers, hidden_dim).
            labels: Integer class labels.
            topology: "cayley" or "fully_connected".
        """
        if not layerwise_list:
            raise ValueError("layerwise_list cannot be empty")

        self.items = [np.asarray(x) for x in layerwise_list]
        self.labels = labels
        self.topology = topology

        num_layers = self.items[0].shape[0]
        self.edge_index, self.num_graph_nodes = build_edge_index(num_layers, topology)
        self.num_real_layers = num_layers

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x = torch.as_tensor(self.items[idx], dtype=torch.float32)  # [L, D]
        num_real = x.size(0)

        # Pad with virtual nodes if graph has more nodes than layers (cayley)
        num_virtual = self.num_graph_nodes - num_real
        if num_virtual > 0:
            virt = torch.zeros(num_virtual, x.size(1), dtype=x.dtype)
            x = torch.cat([x, virt], dim=0)

        from torch_geometric.data import Data as GeomData

        data = GeomData(x=x, edge_index=self.edge_index.clone())
        data.y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        # Node-level mask marking real layer-nodes (True) vs virtual padding
        # (False). PyG concatenates it across the batch; GNNEncoder pools over
        # real nodes only.
        is_real = torch.zeros(x.size(0), dtype=torch.bool)
        is_real[:num_real] = True
        data.is_real = is_real
        return data


class TensorDataset(TorchDataset):
    """
    Dataset for SetEncoder.

    Each sample keeps all layer embeddings as [num_layers, hidden_dim].
    """

    def __init__(self, layerwise_list: List[np.ndarray], labels: List[int]):
        self.X = torch.stack(
            [torch.tensor(x, dtype=torch.float32) for x in layerwise_list]
        )
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def graph_collate(batch):
    """Collate function for GraphDataset -> PyG Batch."""
    from torch_geometric.data import Batch as GeomBatch
    return GeomBatch.from_data_list(batch)


def tensor_collate(batch):
    """Collate function for TensorDataset -> (X_batch, y_batch)."""
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)
