"""
Public aggregator API.

Two aggregator classes:

1. `Aggregator` (via `build_aggregator`):
   Maps (N, num_layers, hidden_in) -> (N, out_dim).
   Each of the N samples is one independent ILSE input.
   Use this for sequence-level classification or per-residue tasks
   (flatten batch*seq_len into N).

2. `PerTokenAggregator` (via `build_per_token_aggregator`):
   Maps (batch, seq_len, num_layers, hidden_in) -> (batch, seq_len, out_dim).
   Builds one Cayley graph per sample with seq_len*num_layers nodes, runs
   GNN message passing, then pools across layers per token to produce
   per-token output. Cayley topology only. Use this when you want cross-token
   information flow through the aggregator (e.g., per-residue protein tasks).

Both are pure nn.Modules. No LLM, no training loop, no caching, no labels.
"""
import types
from typing import Union

import torch
import torch.nn as nn

from .configs import CayleyConfig, FCConfig, SetEncoderConfig
from ._internal.nn_modules import GNNEncoder, SetEncoder, build_gnn_encoder
from ._internal.graph_ops import build_edge_index

AggregatorConfig = Union[CayleyConfig, FCConfig, SetEncoderConfig]


# --------------------------------------------------------------------------- #
# Vectorized batching helpers
#
# Every sample shares the same graph topology, so instead of building per-sample
# PyG `Data` objects and calling `Batch.from_data_list` (a Python loop), we tile
# and offset a single edge_index in one shot and hand GNNEncoder a lightweight
# stand-in for a PyG Batch. This keeps the GNN path fast even for large N.
# --------------------------------------------------------------------------- #

def _tiled_edge_index(edge_index: torch.Tensor, num_nodes: int, n_graphs: int,
                      device: torch.device) -> torch.Tensor:
    """Tile a single-graph edge_index n_graphs times, offsetting each copy."""
    edge_index = edge_index.to(device)
    E = edge_index.size(1)
    offsets = (torch.arange(n_graphs, device=device).repeat_interleave(E) * num_nodes)
    return edge_index.repeat(1, n_graphs) + offsets.unsqueeze(0)


def _batch_assignment(num_nodes: int, n_graphs: int, device: torch.device) -> torch.Tensor:
    """Batch vector [0,0,...,1,1,...,n-1,...] mapping each node to its graph."""
    return torch.arange(n_graphs, device=device).repeat_interleave(num_nodes)


def _real_node_mask(num_real: int, num_nodes: int, n_graphs: int,
                    device: torch.device):
    """Flat bool mask (n_graphs*num_nodes,) marking real nodes; None if no padding."""
    if num_real >= num_nodes:
        return None
    per_graph = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    per_graph[:num_real] = True
    return per_graph.repeat(n_graphs)


def _pyg_like_batch(flat_x, flat_edge_index, batch_idx, n_graphs, is_real):
    """
    Build a minimal stand-in that GNNEncoder.forward() consumes like a PyG
    Batch. GNNEncoder only reads .x / .edge_index / .batch / .num_graphs /
    .is_real -- never the full PyG Batch API -- so a SimpleNamespace suffices
    and avoids constructing real Data/Batch objects.
    """
    return types.SimpleNamespace(
        x=flat_x,
        edge_index=flat_edge_index,
        batch=batch_idx,
        num_graphs=n_graphs,
        is_real=is_real,
    )


# --------------------------------------------------------------------------- #
# Standard aggregator: (N, num_layers, hidden_in) -> (N, out_dim)
# --------------------------------------------------------------------------- #

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
    - For GNN configs (Cayley / FC), all N samples share the same graph
      topology. The batch is constructed via vectorized tensor ops (no Python
      loop over samples), making this efficient even for large N (~4000+
      residues per batch in per-token use cases).
    - For Cayley, the graph may have more nodes than `num_layers` (the
      smallest SL(2, Z_n) with enough nodes). Extra positions are zero-padded
      and participate in message passing, but are excluded from the final pool.
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

            self._encoder = build_gnn_encoder(config, hidden_in)

            topology = "cayley" if isinstance(config, CayleyConfig) else "fully_connected"
            edge_index, num_graph_nodes = build_edge_index(num_layers, topology)
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

        return self._forward_gnn_vectorized(x)

    def _forward_gnn_vectorized(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized GNN forward (see module-level batching helpers)."""
        N = x.size(0)
        num_nodes = self.num_graph_nodes

        # Pad virtual nodes for Cayley if needed.
        num_virtual = num_nodes - self.num_layers
        if num_virtual > 0:
            virt = x.new_zeros(N, num_virtual, self.hidden_in)
            x = torch.cat([x, virt], dim=1)  # (N, num_nodes, hidden_in)

        flat_x = x.reshape(N * num_nodes, self.hidden_in)
        flat_ei = _tiled_edge_index(self.edge_index, num_nodes, N, x.device)
        batch_idx = _batch_assignment(num_nodes, N, x.device)
        is_real = _real_node_mask(self.num_layers, num_nodes, N, x.device)

        batch = _pyg_like_batch(flat_x, flat_ei, batch_idx, N, is_real)
        return self._encoder(batch)


# --------------------------------------------------------------------------- #
# Per-token aggregator: (B, T, L, D) -> (B, T, out_dim)
# --------------------------------------------------------------------------- #

class PerTokenAggregator(nn.Module):
    """
    Aggregator that builds one Cayley graph per sample spanning ALL tokens
    and ALL layers.

    Two output modes:

    - output_mode="per_token" (default):
        Forward: (batch, seq_len, num_layers, hidden_in) -> (batch, seq_len, out_dim)
        After GNN message passing, pools across layers per token.
        Use for per-residue protein tasks.

    - output_mode="sequence":
        Forward: (batch, seq_len, num_layers, hidden_in) -> (batch, out_dim)
        After GNN message passing, global pool across ALL real nodes
        (tokens x layers). Use for sequence-level classification.

    Graph construction:
        For a sample with T tokens and L layers, the graph has T*L nodes
        (plus virtual nodes for Cayley padding). Node indexing:
        node(t, l) = t * L + l  (token t, layer l).

    This mode is Cayley-only. Supports conv_type="gin", "gcn", "gat".

    Cost warning:
        The graph has T*L nodes, and the smallest Cayley graph is sized to fit
        that (|SL(2, Z_n)| ~ n^3). Long sequences produce very large graphs that
        are slow to build and memory-heavy to run; graph_ops emits a warning
        past a few thousand nodes. Keep T modest, or chunk long sequences.

    Notes:
        - The current implementation assumes all samples in a batch share the
          same seq_len (pad to the same length). Padding tokens should be masked
          in the loss, not in the aggregator.
        - The Cayley graph is cached for a given total_nodes count and rebuilt
          only when seq_len changes.
    """

    def __init__(
        self,
        config: CayleyConfig,
        num_layers: int,
        hidden_in: int,
        token_pooling: str = "mean",
        output_mode: str = "per_token",
    ):
        """
        Args:
            config: CayleyConfig (Cayley topology only for multi-token mode).
            num_layers: Number of backbone layers.
            hidden_in: Backbone hidden size.
            token_pooling: How to pool across layers per token after GNN
                (only used when output_mode="per_token"). "mean" or "sum".
            output_mode: "per_token" returns (batch, seq_len, out_dim),
                "sequence" returns (batch, out_dim) via global pooling over
                all real nodes.
        """
        super().__init__()
        if not isinstance(config, CayleyConfig):
            raise TypeError(
                f"PerTokenAggregator only supports CayleyConfig, got {type(config).__name__}"
            )
        if token_pooling not in ("mean", "sum"):
            raise ValueError(f"token_pooling must be 'mean' or 'sum', got {token_pooling!r}")
        if output_mode not in ("per_token", "sequence"):
            raise ValueError(f"output_mode must be 'per_token' or 'sequence', got {output_mode!r}")

        self.config = config
        self.num_layers = num_layers
        self.hidden_in = hidden_in
        self.token_pooling = token_pooling
        self.output_mode = output_mode

        self._encoder = build_gnn_encoder(config, hidden_in)
        self.out_dim = config.hidden_dim

        # Cache for Cayley graph (rebuilt when seq_len changes).
        self._cached_seq_len = None
        self._cached_edge_index = None
        self._cached_num_graph_nodes = None

    def _get_cayley_graph(self, total_nodes: int, device: torch.device):
        """Get or rebuild the Cayley graph for total_nodes."""
        if self._cached_seq_len is not None:
            cached_total = self._cached_seq_len * self.num_layers
            if cached_total == total_nodes and self._cached_edge_index.device == device:
                return self._cached_edge_index, self._cached_num_graph_nodes

        edge_index, num_graph_nodes = build_edge_index(total_nodes, "cayley")
        self._cached_edge_index = edge_index.to(device)
        self._cached_num_graph_nodes = num_graph_nodes
        self._cached_seq_len = total_nodes // self.num_layers
        return self._cached_edge_index, num_graph_nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_layers, hidden_in)
        Returns:
            output_mode="per_token": (batch, seq_len, out_dim)
            output_mode="sequence":  (batch, out_dim)
        """
        if x.dim() != 4:
            raise ValueError(
                f"PerTokenAggregator expects (batch, seq_len, num_layers, hidden_in), "
                f"got shape {tuple(x.shape)}"
            )
        B, T, L, D = x.shape
        if L != self.num_layers:
            raise ValueError(f"Expected num_layers={self.num_layers}, got {L}")
        if D != self.hidden_in:
            raise ValueError(f"Expected hidden_in={self.hidden_in}, got {D}")

        total_real_nodes = T * L
        edge_index, num_graph_nodes = self._get_cayley_graph(total_real_nodes, x.device)
        num_virtual = num_graph_nodes - total_real_nodes

        # Flatten tokens and layers into node dimension: (B, T*L, D)
        # Node ordering: node(t, l) = t * L + l
        x = x.reshape(B, T * L, D)

        # Pad virtual nodes if Cayley graph is larger.
        if num_virtual > 0:
            virt = x.new_zeros(B, num_virtual, D)
            x = torch.cat([x, virt], dim=1)  # (B, num_graph_nodes, D)

        if self.output_mode == "sequence":
            # Global pool over all REAL nodes -> (B, out_dim).
            return self._forward_gnn_global_pool(
                x, edge_index, num_graph_nodes, total_real_nodes, B
            )

        # Per-node features -> pool across layers per token -> (B, T, out_dim).
        node_features = self._forward_gnn_nodes(x, edge_index, num_graph_nodes, B)
        # node_features: (B, num_graph_nodes, hidden_dim)

        # Discard virtual padding nodes.
        node_features = node_features[:, :total_real_nodes, :]  # (B, T*L, hidden_dim)

        # Reshape to (B, T, L, hidden_dim) and pool across layers per token.
        node_features = node_features.reshape(B, T, L, -1)
        if self.token_pooling == "mean":
            return node_features.mean(dim=2)  # (B, T, hidden_dim)
        return node_features.sum(dim=2)

    def _forward_gnn_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_graph_nodes: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Run GNN message passing and return per-node features (no pooling).

        Args:
            x: (B, num_graph_nodes, hidden_in)
            edge_index: (2, E) base Cayley edge_index for one graph
            num_graph_nodes: number of nodes per graph
            batch_size: B

        Returns:
            (B, num_graph_nodes, hidden_dim) per-node features after GNN.
        """
        B = batch_size
        N_nodes = num_graph_nodes

        flat_x = x.reshape(B * N_nodes, -1)
        flat_ei = _tiled_edge_index(edge_index, N_nodes, B, x.device)

        # Run the encoder's projection + conv layers, mirroring
        # GNNEncoder.forward but skipping global pooling (we want per-node out).
        enc = self._encoder
        h = enc.proj_in(flat_x)
        h = enc.act(h)
        h = enc.dropout(h)

        for conv, norm in zip(enc.convs, enc.norms):
            h = conv(h, flat_ei)
            h = norm(h)
            if enc.conv_type in ("gcn", "gat"):
                h = enc.act(h)
            h = enc.dropout(h)

        return h.reshape(B, N_nodes, -1)

    def _forward_gnn_global_pool(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_graph_nodes: int,
        num_real_nodes: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Run GNN message passing with global pooling over all real nodes per
        graph, using GNNEncoder.forward() and the shared vectorized batch.

        Returns:
            (B, out_dim) -- one vector per sample (sequence-level).
        """
        B = batch_size
        N_nodes = num_graph_nodes

        flat_x = x.reshape(B * N_nodes, -1)
        flat_ei = _tiled_edge_index(edge_index, N_nodes, B, x.device)
        batch_idx = _batch_assignment(N_nodes, B, x.device)
        is_real = _real_node_mask(num_real_nodes, N_nodes, B, x.device)

        batch = _pyg_like_batch(flat_x, flat_ei, batch_idx, B, is_real)
        return self._encoder(batch)


# --------------------------------------------------------------------------- #
# Factory functions
# --------------------------------------------------------------------------- #

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
            transformers this is `model.config.num_hidden_layers + 1` (+1 for
            the embedding-layer output). Use `ilse.num_layers_for(model)` to get
            it without hardcoding.
        hidden_in: Backbone hidden size (`model.config.hidden_size`, or
            `ilse.hidden_size_for(model)`).

    Returns:
        Aggregator nn.Module. Forward takes (N, num_layers, hidden_in) and
        returns (N, aggregator.out_dim).

    Example:
        >>> import torch
        >>> from ilse import build_aggregator, CayleyConfig
        >>> agg = build_aggregator(CayleyConfig(), num_layers=25, hidden_in=1024)
        >>> x = torch.randn(8, 25, 1024)
        >>> agg(x).shape
        torch.Size([8, 256])
        >>> agg.out_dim
        256
    """
    return Aggregator(config=config, num_layers=num_layers, hidden_in=hidden_in)


def build_per_token_aggregator(
    config: CayleyConfig,
    num_layers: int,
    hidden_in: int,
    token_pooling: str = "mean",
    output_mode: str = "per_token",
) -> PerTokenAggregator:
    """
    Build a multi-token ILSE aggregator that connects ALL tokens and ALL
    layers in a single Cayley graph per sample.

    Args:
        config: CayleyConfig (Cayley topology only for multi-token mode).
        num_layers: Number of backbone layers.
        hidden_in: Backbone hidden size.
        token_pooling: How to pool across layers per token after GNN message
            passing. "mean" or "sum". Only used when output_mode="per_token".
        output_mode: Controls output shape:
            - "per_token": (batch, seq_len, out_dim) -- pool across layers per
              token. Use for per-residue protein tasks with a frozen PLM.
            - "sequence": (batch, out_dim) -- global pool over all real nodes.
              Use for sequence-level classification.

    Cost warning:
        The graph spans seq_len*num_layers nodes; long sequences make very large
        Cayley graphs (slow to build, memory-heavy). Keep seq_len modest.

    Returns:
        PerTokenAggregator nn.Module. Forward takes
        (batch, seq_len, num_layers, hidden_in).

    Examples:
        Per-token output (per-residue regression on a small window):
        >>> import torch
        >>> from ilse import build_per_token_aggregator, CayleyConfig
        >>> cfg = CayleyConfig(conv_type="gat", gat_heads=4, gnn_layers=2)
        >>> agg = build_per_token_aggregator(cfg, num_layers=13, hidden_in=480)
        >>> x = torch.randn(2, 16, 13, 480)
        >>> agg(x).shape
        torch.Size([2, 16, 256])

        Sequence-level output (short-text classification):
        >>> agg = build_per_token_aggregator(cfg, num_layers=13, hidden_in=480,
        ...                                  output_mode="sequence")
        >>> x = torch.randn(4, 16, 13, 480)
        >>> agg(x).shape
        torch.Size([4, 256])
    """
    return PerTokenAggregator(
        config=config,
        num_layers=num_layers,
        hidden_in=hidden_in,
        token_pooling=token_pooling,
        output_mode=output_mode,
    )
