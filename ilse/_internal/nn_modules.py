"""
Core neural network modules for ILSE encoders.

Three encoder families:
- GNNEncoder: GIN/GCN/GAT message passing over layer-graphs (used by Cayley and FC)
- SetEncoder: DeepSet-style phi -> pool -> rho (permutation-invariant)
- ClassificationHead: simple linear head on top of any encoder
"""
import torch
import torch.nn as nn


def _import_pyg():
    """Lazy import for PyTorch Geometric (only needed by GNN encoders)."""
    try:
        from torch_geometric.nn import (
            GINConv, GCNConv, GATConv,
            global_mean_pool, global_add_pool,
        )
        return GINConv, GCNConv, GATConv, global_mean_pool, global_add_pool
    except ImportError:
        raise ImportError(
            "torch-geometric is required for GNN encoders (Cayley, FC). "
            "Install it: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
        )


class GNNEncoder(nn.Module):
    """
    GNN encoder for layer-graph inputs.

    Supports GINConv (conv_type="gin"), GCNConv (conv_type="gcn"), and
    GATConv (conv_type="gat"). Used by both Cayley and FC topologies.

    Pooling is over the REAL layer-nodes only. Cayley graphs pad up to the
    nearest SL(2, Z_n) size with virtual nodes; those participate in message
    passing but are excluded from the final pool (via `batch.is_real`), so the
    aggregated vector reflects only the actual LLM layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        gnn_layers: int = 1,
        gin_mlp_layers: int = 1,
        conv_type: str = "gin",
        pooling: str = "mean",
        dropout: float = 0.1,
        gat_heads: int = 4,
    ):
        super().__init__()
        if conv_type not in ("gin", "gcn", "gat"):
            raise ValueError(f"conv_type must be 'gin', 'gcn', or 'gat', got {conv_type!r}")
        if pooling not in ("mean", "sum"):
            raise ValueError(f"pooling must be 'mean' or 'sum', got {pooling!r}")

        self.conv_type = conv_type
        self.pooling = pooling
        self.out_dim = hidden_dim

        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        GINConv, GCNConv, GATConv, global_mean_pool, global_add_pool = _import_pyg()
        # Cache the pooling op once instead of importing PyG on every forward.
        self._pool_fn = global_mean_pool if pooling == "mean" else global_add_pool

        self.convs = nn.ModuleList()
        for _ in range(gnn_layers):
            if conv_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == "gat":
                # concat=False: each head produces hidden_dim features,
                # outputs are averaged. No divisibility constraint on hidden_dim.
                self.convs.append(GATConv(
                    hidden_dim, hidden_dim,
                    heads=gat_heads, concat=False, dropout=dropout,
                ))
            else:
                mlp_modules = []
                for _ in range(gin_mlp_layers):
                    mlp_modules.append(nn.Linear(hidden_dim, hidden_dim))
                    mlp_modules.append(nn.ReLU())
                mlp = nn.Sequential(*mlp_modules)
                self.convs.append(GINConv(mlp))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(gnn_layers)])

    def forward(self, batch) -> torch.Tensor:
        x = self.proj_in(batch.x)
        x = self.act(x)
        x = self.dropout(x)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, batch.edge_index)
            x = norm(x)
            if self.conv_type in ("gcn", "gat"):
                x = self.act(x)
            x = self.dropout(x)

        # Pool over real nodes only. Virtual padding nodes (Cayley) carry no
        # LLM layer and are dropped here even though they took part in message
        # passing above.
        node_batch = batch.batch
        is_real = getattr(batch, "is_real", None)
        if is_real is not None:
            x = x[is_real]
            node_batch = node_batch[is_real]

        size = getattr(batch, "num_graphs", None)
        return self._pool_fn(x, node_batch, size=size)


def build_gnn_encoder(config, in_dim: int) -> GNNEncoder:
    """
    Construct a GNNEncoder from a Cayley/FC config.

    Single source of truth for turning a config into an encoder, shared by
    ILSEClassifier and the aggregator factories so they cannot drift apart
    (e.g. all paths pass gat_heads).
    """
    return GNNEncoder(
        in_dim=in_dim,
        hidden_dim=config.hidden_dim,
        gnn_layers=config.gnn_layers,
        gin_mlp_layers=config.gin_mlp_layers if config.conv_type == "gin" else 0,
        conv_type=config.conv_type,
        pooling=config.pooling,
        dropout=config.dropout,
        gat_heads=getattr(config, "gat_heads", 4),
    )


class SetEncoder(nn.Module):
    """
    DeepSet encoder: phi(per-layer MLP) -> pool -> rho(post-pool MLP).

    Treats LLM layers as an unordered set (permutation-invariant).
    """

    def __init__(
        self,
        num_layers: int,
        layer_dim: int,
        hidden_dim: int = 256,
        pre_pooling_layers: int = 1,
        post_pooling_layers: int = 1,
        pooling: str = "sum",
        dropout: float = 0.2,
    ):
        super().__init__()
        if pooling not in ("mean", "sum"):
            raise ValueError(f"pooling must be 'mean' or 'sum', got {pooling!r}")

        self.num_layers = num_layers
        self.layer_dim = layer_dim
        self.pooling = pooling

        # Pre-pooling MLP (phi): applied to each layer independently
        if pre_pooling_layers > 0:
            layers = []
            layers.append(nn.Linear(layer_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(pre_pooling_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            self.phi = nn.Sequential(*layers)
            pool_in_dim = hidden_dim
        else:
            self.phi = None
            pool_in_dim = layer_dim

        # Post-pooling MLP (rho): applied after aggregation
        if post_pooling_layers > 0:
            layers = []
            layers.append(nn.Linear(pool_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(post_pooling_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            self.rho = nn.Sequential(*layers)
            self.out_dim = hidden_dim
        else:
            self.rho = None
            self.out_dim = pool_in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_layers, layer_dim]
        Returns:
            [batch, out_dim]
        """
        B = x.size(0)

        if self.phi is not None:
            x_flat = x.view(B * self.num_layers, self.layer_dim)
            x = self.phi(x_flat).view(B, self.num_layers, -1)

        if self.pooling == "mean":
            pooled = x.mean(dim=1)
        else:
            pooled = x.sum(dim=1)

        if self.rho is not None:
            return self.rho(pooled)
        return pooled


class ClassificationHead(nn.Module):
    """Linear classification head on top of an encoder."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)
