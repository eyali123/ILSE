"""
Public aggregator API.

`build_aggregator` returns a pure `nn.Module` that maps
    (N, num_layers, hidden_in) -> (N, out_dim)

It contains no LLM, no training loop, no caching, and no labels.
The caller owns the backbone, the loss, the optimizer, and the training loop.

This is the integration point for projects that want to plug ILSE into their
own fine-tuning pipeline (e.g., per-token regression, ordinal heads, custom
losses). For simple sklearn-style classification, see `ILSEClassifier`.
"""
from typing import Union

import torch
import torch.nn as nn

from .configs import CayleyConfig, FCConfig, SetEncoderConfig
from ._internal.nn_modules import GNNEncoder, SetEncoder
from ._internal.graph_ops import build_edge_index

AggregatorConfig = Union[CayleyConfig, FCConfig, SetEncoderConfig]


class Aggregator(nn.Module):
    """
    Pure nn.Module wrapping an ILSE encoder.

    Forward signature:
        x: (N, num_layers, hidden_in)  -- dense tensor of per-sample layer stacks
        returns: (N, out_dim)          -- one aggregated vector per sample

    The caller is responsible for:
    - Running the (frozen) backbone and stacking its hidden states into `x`.
    - Attaching a task-specific head on top of the returned vectors.
    - The training loop, loss, and optimizer.

    Notes:
    - For GNN configs (Cayley / FC), each of the N samples becomes one graph
      with `num_layers` nodes. Batch construction is handled internally; the
      public API stays a dense tensor.
    - For Cayley, the graph may have more nodes than `num_layers` (the smallest
      SL(2, Z_n) with enough nodes). Extra positions are padded with zeros.
    - `out_dim` is exposed so the caller can size their head correctly.
    """

    def __init__(
        self,
        config: AggregatorConfig,
        num_layers: int,
        hidden_in: int,
    ):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.hidden_in = hidden_in
        self._is_gnn = not isinstance(config, SetEncoderConfig)

        if self._is_gnn:
            if not isinstance(config, (CayleyConfig, FCConfig)):
                raise TypeError(
                    f"GNN path expects CayleyConfig or FCConfig, got {type(config).__name__}"
                )

            self._encoder = GNNEncoder(
                in_dim=hidden_in,
                hidden_dim=config.hidden_dim,
                gnn_layers=config.gnn_layers,
                gin_mlp_layers=config.gin_mlp_layers if config.conv_type == "gin" else 0,
                conv_type=config.conv_type,
                pooling=config.pooling,
                dropout=config.dropout,
            )

            topology = "cayley" if isinstance(config, CayleyConfig) else "fully_connected"
            edge_index, num_graph_nodes = build_edge_index(num_layers, topology)
            # Register as buffer so .to(device) moves it with the module.
            self.register_buffer("edge_index", edge_index, persistent=False)
            self.num_graph_nodes = num_graph_nodes
            self.out_dim = config.hidden_dim

        else:
            self._encoder = SetEncoder(
                num_layers=num_layers,
                layer_dim=hidden_in,
                hidden_dim=config.hidden_dim,
                pre_pooling_layers=config.pre_pooling_layers,
                post_pooling_layers=config.post_pooling_layers,
                pooling=config.pooling,
                dropout=config.dropout,
            )
            self.out_dim = self._encoder.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, num_layers, hidden_in)
        Returns:
            (N, out_dim)
        """
        if x.dim() != 3:
            raise ValueError(
                f"Aggregator expects (N, num_layers, hidden_in), got shape {tuple(x.shape)}"
            )
        if x.size(1) != self.num_layers:
            raise ValueError(
                f"Expected num_layers={self.num_layers} on dim 1, got {x.size(1)}"
            )
        if x.size(2) != self.hidden_in:
            raise ValueError(
                f"Expected hidden_in={self.hidden_in} on dim 2, got {x.size(2)}"
            )

        if not self._is_gnn:
            return self._encoder(x)

        # GNN path: convert dense tensor to a PyG Batch.
        # Lazy import so the SetEncoder path doesn't require torch_geometric.
        from torch_geometric.data import Batch, Data

        N = x.size(0)

        # Pad virtual nodes for Cayley if graph has more nodes than layers.
        num_virtual = self.num_graph_nodes - self.num_layers
        if num_virtual > 0:
            virt = x.new_zeros(N, num_virtual, self.hidden_in)
            x = torch.cat([x, virt], dim=1)  # (N, num_graph_nodes, hidden_in)

        # One Data per sample. edge_index is cloned per Data to match the
        # pattern used by GraphDataset and avoid aliasing during PyG batching.
        data_list = [
            Data(x=x[i], edge_index=self.edge_index.clone()) for i in range(N)
        ]
        batch = Batch.from_data_list(data_list)
        return self._encoder(batch)


def build_aggregator(
    config: AggregatorConfig,
    num_layers: int,
    hidden_in: int,
) -> Aggregator:
    """
    Build an ILSE aggregator as a plain nn.Module.

    Args:
        config: One of CayleyConfig, FCConfig, SetEncoderConfig. Determines the
            encoder family and its hyperparameters. Training-loop fields on the
            config (lr, weight_decay, batch_size, epochs) are ignored here --
            the caller owns the training loop.
        num_layers: Number of backbone layers to aggregate. For HuggingFace
            transformers, this is typically `model.config.num_hidden_layers + 1`
            (+1 for the embedding layer output).
        hidden_in: Backbone hidden size. For HuggingFace transformers, this is
            `model.config.hidden_size`.

    Returns:
        Aggregator nn.Module. Forward takes (N, num_layers, hidden_in) and
        returns (N, aggregator.out_dim).

    Example:
        >>> from ilse import build_aggregator, CayleyConfig
        >>> agg = build_aggregator(CayleyConfig(), num_layers=37, hidden_in=2560)
        >>> import torch
        >>> x = torch.randn(8, 37, 2560)
        >>> y = agg(x)
        >>> y.shape
        torch.Size([8, 256])
        >>> agg.out_dim
        256
    """
    return Aggregator(config=config, num_layers=num_layers, hidden_in=hidden_in)
