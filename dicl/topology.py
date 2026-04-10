"""
dicl/topology.py
=================
Communication graph topologies and consensus aggregation functions.

Topologies
----------
  fully_connected  — every agent communicates with every other
  ring             — agents arranged in a cycle
  star             — one hub agent communicates with all others
  sparse_random    — Erdős–Rényi random graph (p=0.4), seeded

Consensus functions
-------------------
  arithmetic  — unweighted mean of neighbourhood probabilities
  weighted    — weighted mean (uniform if no weights supplied)
  geometric   — log-space mean followed by softmax normalisation

Usage
-----
    from dicl.topology import TOPOLOGIES, aggregate_all

    A      = TOPOLOGIES["ring"](K)
    bar    = aggregate_all(prob_list, A, consensus="arithmetic")
"""

import numpy as np


# =============================================================================
# Graph adjacency matrices
# =============================================================================

def _topo_fc(K: int) -> np.ndarray:
    """Fully-connected: all agents are neighbours."""
    return np.ones((K, K))


def _topo_ring(K: int) -> np.ndarray:
    """Ring: each agent talks to its two cyclic neighbours (and itself)."""
    A = np.eye(K)
    for k in range(K):
        A[k, (k - 1) % K] = 1
        A[k, (k + 1) % K] = 1
    return A


def _topo_star(K: int) -> np.ndarray:
    """Star: agent 0 is the hub; all others connect only through agent 0."""
    A = np.zeros((K, K))
    A[0, :] = 1
    A[:, 0] = 1
    A[np.diag_indices(K)] = 1
    return A


def _topo_sparse(K: int, p: float = 0.4, seed: int = 0) -> np.ndarray:
    """
    Sparse random graph (Erdős–Rényi, p=0.4).

    Each edge (i,j) with i<j is included independently with probability p.
    Isolated nodes are connected to agent 0 to guarantee connectivity.
    """
    rng = np.random.RandomState(seed)
    A   = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            if rng.rand() < p:
                A[i, j] = A[j, i] = 1
    for k in range(1, K):
        if A[k].sum() == 1:
            A[k, 0] = A[0, k] = 1
    return A


TOPOLOGIES: dict = {
    "fully_connected": _topo_fc,
    "ring":            _topo_ring,
    "star":            _topo_star,
    "sparse_random":   _topo_sparse,
}


# =============================================================================
# Consensus aggregation
# =============================================================================

def _cons_arith(
    prob_list: list[np.ndarray],
    row: np.ndarray,
) -> np.ndarray:
    """Arithmetic (unweighted) mean over neighbourhood."""
    nb = np.where(row > 0)[0]
    return np.stack([prob_list[j] for j in nb]).mean(0)


def _cons_weighted(
    prob_list: list[np.ndarray],
    row: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted mean over neighbourhood (uniform if weights is None)."""
    nb = np.where(row > 0)[0]
    w  = (
        np.ones(len(nb)) / len(nb)
        if weights is None
        else weights[nb] / weights[nb].sum()
    )
    return (np.stack([prob_list[j] for j in nb]) * w[:, None, None]).sum(0)


def _cons_geo(
    prob_list: list[np.ndarray],
    row: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Geometric mean via log-space averaging followed by softmax normalisation.

    Equivalent to the product-of-experts ensemble, numerically stable via
    max-subtraction before exp.
    """
    nb = np.where(row > 0)[0]
    la = np.stack([np.log(prob_list[j] + eps) for j in nb]).mean(0)
    la -= la.max(1, keepdims=True)
    e   = np.exp(la)
    return e / e.sum(1, keepdims=True)


CONSENSUS_FNS: dict = {
    "arithmetic": _cons_arith,
    "weighted":   _cons_weighted,
    "geometric":  _cons_geo,
}


def aggregate_all(
    prob_list: list[np.ndarray],
    A: np.ndarray,
    consensus: str = "arithmetic",
) -> list[np.ndarray]:
    """
    Apply the consensus function for every agent's neighbourhood.

    Parameters
    ----------
    prob_list : per-agent probability arrays  [K × (N, C)]
    A         : adjacency matrix              (K, K)
    consensus : key into CONSENSUS_FNS

    Returns
    -------
    List of K aggregated probability arrays, one per agent.
    """
    fn = CONSENSUS_FNS[consensus]
    return [fn(prob_list, A[k]) for k in range(len(prob_list))]
