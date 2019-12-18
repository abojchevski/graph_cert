"""
Implementation of the method proposed in the paper:
'Certifiable Robustness to Graph Perturbations'
Aleksandar Bojchevski and Stephan Günnemann, NeurIPS 2019

Copyright (C) owned by the authors, 2019
"""
import numba
import numpy as np
import scipy.sparse as sp


@numba.jit(nopython=True)
def _top_k(indices, indptr, data, k_per_row):
    """

    Parameters
    ----------
    indices: np.ndarray, shape [n_edges]
        Indices of a sparse matrix.
    indptr: np.ndarray, shape [n+1]
        Index pointers of a sparse matrix.
    data: np.ndarray, shape [n_edges]
        Data of a sparse matrix.
    k_per_row: np.ndarray, shape [n]
        Number of top_k elements for each row.
    Returns
    -------
    top_k_idx: list
        List of the indices of the top_k elements for each row.
    """
    n = len(indptr) - 1
    top_k_idx = []
    for i in range(n):
        cur_top_k = k_per_row[i]
        if cur_top_k > 0:
            cur_indices = indices[indptr[i]:indptr[i + 1]]
            cur_data = data[indptr[i]:indptr[i + 1]]
            # top_k = cur_indices[np.argpartition(cur_data, -cur_budget)[-cur_budget:]]
            top_k = cur_indices[cur_data.argsort()[-cur_top_k:]]
            top_k_idx.append(top_k)

    return top_k_idx


def top_k_numba(x, k_per_row):
    """
    Returns the indices of the top_k element per row for a sparse matrix.
    Considers only the non-zero entries.

    Parameters
    ----------
    x : sp.spmatrix, shape [n, n]
        Data matrix.
    k_per_row : np.ndarray, shape [n]
        Number of top_k elements for each row.

    Returns
    -------
    top_k_per_row : np.ndarray, shape [?, 2]
        The 2D indices of the top_k elements per row.
    """
    # make sure that k_per_row does not exceed the number of non-zero elements per row
    k_per_row = np.minimum(k_per_row, (x != 0).sum(1).A1)
    n = x.shape[0]
    row_idx = np.repeat(np.arange(n), k_per_row)

    col_idx = _top_k(x.indices, x.indptr, x.data, k_per_row)
    col_idx = np.concatenate(col_idx)

    top_k_per_row = np.column_stack((row_idx, col_idx))

    return top_k_per_row


def flip_edges(adj, edges):
    """
    Flip the edges in the graph (A_ij=1 becomes A_ij=0, and A_ij=0 becomes A_ij=1).

    Parameters
    ----------
    adj : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    edges : np.ndarray, shape [?, 2]
        Edges to flip.
    Returns
    -------
    adj_flipped : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix with flipped edges.
    """
    adj_flipped = adj.copy().tolil()
    if len(edges) > 0:
        adj_flipped[edges[:, 0], edges[:, 1]] = 1 - adj[edges[:, 0], edges[:, 1]]
    return adj_flipped


def propagation_matrix(adj, alpha=0.85, sigma=1):
    """
    Computes the propagation matrix  (1-alpha)(I - alpha D^{-sigma} A D^{sigma-1})^{-1}.

    Parameters
    ----------
    adj : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    alpha : float
        (1-alpha) is the teleport probability.
    sigma
        Hyper-parameter controlling the propagation style.
        Set sigma=1 to obtain the PPR matrix.
    Returns
    -------
    prop_matrix : np.ndarray, shape [n, n]
        Propagation matrix.

    """
    deg = adj.sum(1).A1
    deg_min_sig = sp.diags(np.power(deg, -sigma))
    deg_sig_min = sp.diags(np.power(deg, sigma - 1))

    n = adj.shape[0]
    pre_inv = sp.eye(n) - alpha * deg_min_sig @ adj @ deg_sig_min

    prop_matrix = (1 - alpha) * np.linalg.inv(pre_inv.toarray())
    return prop_matrix


def correction_term(adj, opt_fragile, fragile):
    """
    Computes correction term needed to map x_v to ppr_v.

    Parameters
    ----------
    adj : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    opt_fragile : np.ndarray, shape [?, 2]
        Optimal fragile edges.
    fragile : np.ndarray, shape [?, 2]
        Fragile edges that are under our control.

    Returns
    -------
    correction : np.ndarray, shape [n]
        Correction term.
    """
    n = adj.shape[0]
    if len(opt_fragile) > 0:
        adj_all = adj + edges_to_sparse(fragile, n)
        adj_all[adj_all != 0] = 1
        deg_all = adj_all.sum(1).A1

        g_chosen = edges_to_sparse(opt_fragile, n, 1 - 2 * adj[opt_fragile[:, 0], opt_fragile[:, 1]].A1)
        n_removed = -g_chosen.multiply(g_chosen == -1).sum(1).A1
        n_added = g_chosen.multiply(g_chosen == 1).sum(1).A1
        n_to_add = edges_to_sparse(fragile, n, 1 - adj[fragile[:, 0], fragile[:, 1]].A1).sum(1).A1
        correction = 1 - (n_removed + (n_to_add - n_added)) / deg_all
    else:
        correction = np.ones(n)

    return correction


def topic_sensitive_pagerank(adj, alpha, teleport):
    """
    Computes the topic-sensitive PageRank vector.

    Parameters
    ----------
    adj : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    alpha : float
        (1-alpha) teleport[v] is the probability to teleport to node v.
    teleport : np.ndarray, shape [n]
        Teleport vector.

    Returns
    -------
    ppr : np.ndarray, shape [n]
        PageRank vector.
    """
    assert np.isclose(teleport.sum(), 1)

    n = adj.shape[0]
    trans = sp.diags(1 / adj.sum(1).A1) @ adj.tocsr()

    # gets one row from the PPR matrix (since we transpose the transition matrix)
    ppr = sp.linalg.gmres(sp.eye(n) - alpha * trans.T, teleport)[0] * (1 - alpha)

    return ppr


def edges_to_sparse(edges, num_nodes, weights=None):
    """Create a sparse adjacency matrix from an array of edge indices and (optionally) values.

    :param edges: array-like, shape [num_edges, 2]
        Array with each row storing indices of an edge as (u, v).
    :param num_nodes: int
        Number of nodes in the resulting graph.
    :param weights: array_like, shape [num_edges], optional, default None
        Weights of the edges. If None, all edges weights are set to 1.
    :return: sp.csr_matrix
        Adjacency matrix in CSR format.
    """
    if weights is None:
        weights = np.ones(edges.shape[0])

    return sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes)).tocsr()


def get_fragile(adj, addrem):
    """
    Generate a set of fragile edges corresponding to different threat models and scenarios.

    Parameters
    ----------
    adj : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    addrem : string
        'rem' specifies an attacker that can only remove edges, i.e. fragile edges are existing edges in the graph,
        'addrem' specifies an attacker that can both add and remove edges.
    both_dir : bool
        Whether to make both directions (i, j) and (j, i) fragile.

    Returns
    -------
    fragile : np.ndarray, shape [?, 2]
        Set of fragile edges.
    """
    n = adj.shape[0]

    mst = sp.csgraph.minimum_spanning_tree(adj)
    mst = mst + mst.T

    if addrem == 'rem':
        fragile = np.column_stack((adj - mst).nonzero())
    elif addrem == 'addrem':
        fragile_rem = np.column_stack((adj - mst).nonzero())
        fragile_add = np.column_stack(np.ones((n, n)).nonzero())
        fragile_add = fragile_add[adj[fragile_add[:, 0], fragile_add[:, 1]].A1 == 0]
        fragile_add = fragile_add[fragile_add[:, 0] != fragile_add[:, 1]]
        fragile = np.row_stack((fragile_add, fragile_rem))
    else:
        raise ValueError('addrem not set correctly.')

    return fragile
