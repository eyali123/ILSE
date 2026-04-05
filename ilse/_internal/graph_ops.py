"""
Graph construction utilities for ILSE encoders.

Supports two topologies:
- Fully-Connected (FC): all layer-nodes connected to each other
- Cayley (SL2): algebraic Cayley graph from SL(2, Z_n), with virtual nodes for padding
"""
import numpy as np
import torch
from collections import deque


def get_cayley_graph(n: int) -> np.ndarray:
    """
    Build the Cayley graph of SL(2, Z_n) with 4 generators.

    For n=3: 24 nodes, n=4: 48 nodes, n=5: 120 nodes.

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

    while queue and len(nodes) < 1000:
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


def find_minimal_cayley_n(target_nodes: int, max_n: int = 10) -> tuple:
    """
    Find minimal n such that |SL(2, Z_n)| >= target_nodes.

    Returns:
        (n, actual_nodes): The n value and the number of nodes in SL(2, Z_n).
    """
    for n in range(2, max_n + 1):
        edge_index = get_cayley_graph(n)
        actual_nodes = int(edge_index.max()) + 1
        if actual_nodes >= target_nodes:
            return n, actual_nodes
    raise ValueError(
        f"Could not find Cayley graph with >= {target_nodes} nodes (tried up to n={max_n})."
    )


def build_edge_index(num_nodes: int, topology: str) -> torch.LongTensor:
    """
    Build edge_index [2, E] for a graph over LLM layers.

    Args:
        num_nodes: Number of layer-nodes.
        topology: "fully_connected" or "cayley".

    Returns:
        edge_index: torch.LongTensor of shape [2, E].
        num_graph_nodes: Actual number of nodes in the graph (may be > num_nodes for cayley due to virtual nodes).
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
