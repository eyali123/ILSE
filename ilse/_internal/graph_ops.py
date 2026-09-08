"""
Graph construction utilities for ILSE encoders.

Supports two topologies:
- Fully-Connected (FC): all layer-nodes connected to each other
- Cayley (SL2): algebraic Cayley graph from SL(2, Z_n), with virtual nodes for padding
"""
import warnings
from collections import deque

import numpy as np
import torch

# Building a Cayley graph much larger than this triggers a warning: the graph
# is materialized with a pure-Python BFS and stored densely, so very large node
# counts (e.g. per-token graphs over long sequences) can be slow to build and
# heavy on memory/compute for a given GPU. It is not an error -- just a heads-up.
_LARGE_GRAPH_WARN_NODES = 2000


def _distinct_prime_factors(n: int):
    """Distinct prime factors of n (trial division)."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def _sl2_zn_order(n: int) -> int:
    """
    Order of the group SL(2, Z_n): n^3 * prod_{p | n} (1 - 1/p^2).

    Matches the number of nodes enumerated by get_cayley_graph(n), since the
    four elementary generators generate the whole group. Computed analytically
    so find_minimal_cayley_n can size the graph without building it.
    """
    order = n ** 3
    for p in _distinct_prime_factors(n):
        order = order * (p * p - 1) // (p * p)
    return order


def get_cayley_graph(n: int) -> np.ndarray:
    """
    Build the Cayley graph of SL(2, Z_n) with 4 generators.

    For n=3: 24 nodes, n=4: 48 nodes, n=5: 120 nodes.

    Note: the whole group is enumerated with a pure-Python BFS. This is cheap
    for the node counts used in layer aggregation (tens of nodes) but grows
    with |SL(2, Z_n)| ~ n^3, so large n (e.g. per-token graphs) can be slow.

    Returns:
        edge_index: numpy array of shape [2, E] (undirected edges).
    """
    generators = np.array([
        [[1, 1], [0, 1]],
        [[1, n - 1], [0, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [n - 1, 1]],
    ], dtype=np.int32)

    def mat_mult_mod(a, b, mod):
        result = np.zeros((2, 2), dtype=np.int32)
        for i in range(2):
            for j in range(2):
                result[i, j] = (a[i, 0] * b[0, j] + a[i, 1] * b[1, j]) % mod
        return result

    def mat_to_tuple(mat):
        return tuple(mat.flatten())

    identity = np.array([[1, 0], [0, 1]], dtype=np.int32)
    queue = deque([identity])
    nodes = {mat_to_tuple(identity): 0}
    node_list = [identity]

    # Enumerate the full group. The generating set is closed under the group,
    # so BFS reaches every element; no artificial node cap (removing the cap
    # is what lets large per-token graphs build at all).
    while queue:
        current = queue.popleft()
        for gen in generators:
            new_matrix = mat_mult_mod(current, gen, n)
            new_tuple = mat_to_tuple(new_matrix)
            if new_tuple not in nodes:
                nodes[new_tuple] = len(nodes)
                node_list.append(new_matrix)
                queue.append(new_matrix)

    num_nodes = len(nodes)
    src_list, dst_list = [], []
    for node_idx in range(num_nodes):
        current_matrix = node_list[node_idx]
        for gen in generators:
            neighbor_matrix = mat_mult_mod(current_matrix, gen, n)
            neighbor_idx = nodes[mat_to_tuple(neighbor_matrix)]
            src_list.append(node_idx)
            dst_list.append(neighbor_idx)
            src_list.append(neighbor_idx)
            dst_list.append(node_idx)

    return np.array([src_list, dst_list], dtype=np.int64)


def find_minimal_cayley_n(target_nodes: int, max_n: int = 128) -> tuple:
    """
    Find minimal n such that |SL(2, Z_n)| >= target_nodes.

    The group order is computed analytically (no graph is built here), so this
    is cheap even for large targets. A warning is emitted if the chosen graph
    is large enough that building it may be slow / memory-heavy.

    Returns:
        (n, actual_nodes): The n value and the number of nodes in SL(2, Z_n).
    """
    for n in range(2, max_n + 1):
        actual_nodes = _sl2_zn_order(n)
        if actual_nodes >= target_nodes:
            if actual_nodes > _LARGE_GRAPH_WARN_NODES:
                warnings.warn(
                    f"Cayley graph for target_nodes={target_nodes} needs "
                    f"n={n} -> {actual_nodes} nodes. This is a large graph: "
                    f"building it (pure-Python BFS) may be slow, and running a "
                    f"GNN over it may exceed memory on smaller GPUs.",
                    stacklevel=2,
                )
            return n, actual_nodes
    raise ValueError(
        f"Could not find Cayley graph with >= {target_nodes} nodes "
        f"(tried up to n={max_n}). Increase max_n if you really need this many."
    )


def build_edge_index(num_nodes: int, topology: str) -> tuple:
    """
    Build edge_index [2, E] for a graph over LLM layers.

    Args:
        num_nodes: Number of layer-nodes.
        topology: "fully_connected" or "cayley".

    Returns:
        (edge_index, num_graph_nodes) where edge_index is a torch.LongTensor of
        shape [2, E] and num_graph_nodes is the actual node count (may be
        > num_nodes for cayley due to virtual-node padding).
    """
    if topology == "fully_connected":
        comb = torch.combinations(torch.arange(num_nodes), r=2).t()
        edge = torch.cat([comb, comb.flip(0)], dim=1)
        return edge.long(), num_nodes

    elif topology == "cayley":
        n, cayley_nodes = find_minimal_cayley_n(num_nodes)
        edge_index_np = get_cayley_graph(n)
        edge = torch.tensor(edge_index_np, dtype=torch.long)
        return edge, cayley_nodes

    else:
        raise ValueError(f"Unknown topology: {topology!r}. Use 'fully_connected' or 'cayley'.")
