"""Graph construction: Cayley/FC sizing, analytic order, cap removal, warnings."""
import pytest
import torch

from ilse._internal.graph_ops import (
    get_cayley_graph,
    _sl2_zn_order,
    find_minimal_cayley_n,
    build_edge_index,
    _LARGE_GRAPH_WARN_NODES,
)


def test_analytic_order_matches_bfs():
    # find_minimal_cayley_n now sizes graphs analytically; that count MUST equal
    # what get_cayley_graph actually enumerates, or padding would be wrong.
    for n in range(2, 8):
        bfs_nodes = int(get_cayley_graph(n).max()) + 1
        assert bfs_nodes == _sl2_zn_order(n), n


def test_standard_layer_sizes_unchanged():
    # 25/33/37 layers -> smallest SL(2,Z_n) is n=4 (48 nodes), as before.
    for L in (25, 33, 37):
        n, nodes = find_minimal_cayley_n(L)
        assert nodes >= L and nodes == 48


def test_cap_removed_large_graph_builds():
    # Old code capped BFS at <1000 nodes; n=11 has 1320 and must build now.
    g = get_cayley_graph(11)
    assert int(g.max()) + 1 == 1320


def test_large_target_warns_instead_of_crashing():
    target = _LARGE_GRAPH_WARN_NODES + 500
    with pytest.warns(UserWarning):
        n, nodes = find_minimal_cayley_n(target)
    assert nodes >= target


def test_build_edge_index_fully_connected():
    ei, nodes = build_edge_index(5, "fully_connected")
    assert nodes == 5
    assert ei.dtype == torch.long
    assert ei.shape == (2, 5 * 4)  # C(5,2)=10 undirected -> 20 directed


def test_build_edge_index_cayley():
    ei, nodes = build_edge_index(7, "cayley")
    assert nodes == 24  # n=3
    assert ei.dtype == torch.long
    assert int(ei.max()) < nodes


def test_build_edge_index_bad_topology():
    with pytest.raises(ValueError):
        build_edge_index(5, "banana")
