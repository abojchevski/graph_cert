"""
Implementation of the method proposed in the paper:
'Certifiable Robustness to Graph Perturbations'
Aleksandar Bojchevski and Stephan Günnemann, NeurIPS 2019

Copyright (C) owned by the authors, 2019
"""
import gust
import numpy as np
import cvxpy as cp
import scipy.sparse as sp

import warnings
from joblib import Parallel, delayed

from graph_cert.utils import *


def policy_iteration(adj, alpha, fragile, local_budget, reward, teleport, max_iter=1000):
    """
    Performs policy iteration to find the set of fragile edges to flip that maximize (r^T pi),
    where pi is the personalized PageRank of the perturbed graph.

    Parameters
    ----------
    adj : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    alpha : float
        (1-alpha) teleport[v] is the probability to teleport to node v.
    fragile : np.ndarray, shape [?, 2]
        Fragile edges that are under our control.
    local_budget : np.ndarray, shape [n]
        Maximum number of local flips per node.
    reward : np.ndarray, shape [n]
        Reward vector.
    teleport : np.ndarray, shape [n]
        Teleport vector.
        Only used to compute the objective value. Not needed for optimization.
    max_iter : int
        Maximum number of policy iterations.
    Returns
    -------
    opt_fragile : np.ndarray, shape [?, 2]
        Optimal fragile edges.
    obj_value : float
        Optimal objective value.
    """
    n = adj.shape[0]

    cur_fragile = np.array([])
    cur_obj_value = np.inf
    prev_fragile = np.array([[0, 0]])
    max_obj_value = -np.inf

    # if the budget is a scalar set the same budget for all nodes
    if not isinstance(local_budget, np.ndarray):
        local_budget = np.repeat(local_budget, n)

    # does standard value iteration
    for it in range(max_iter):
        adj_flipped = flip_edges(adj, cur_fragile)

        # compute the mean reward before teleportation
        trans_flipped = sp.diags(1 / adj_flipped.sum(1).A1) @ adj_flipped
        mean_reward = sp.linalg.gmres(sp.eye(n) - alpha * trans_flipped, reward)[0]

        # compute the change in the mean reward
        vi = mean_reward[fragile[:, 0]]
        vj = mean_reward[fragile[:, 1]]
        ri = reward[fragile[:, 0]]
        change = vj - ((vi - ri) / alpha)

        # +1 if we are adding a node, -1 if we are removing a node
        add_rem_multiplier = 1 - 2 * adj[fragile[:, 0], fragile[:, 1]].A1
        change = change * add_rem_multiplier

        # only consider the ones that improve our objective function
        improve = change > 0
        frag = gust.edges_to_sparse(fragile[improve], n, change[improve])
        # select the top_k fragile edges
        cur_fragile = top_k_numba(frag, local_budget)

        # compute the objective value
        cur_obj_value = mean_reward @ teleport * (1 - alpha)

        # check for convergence
        edges_are_same = (gust.edges_to_sparse(prev_fragile, n) - gust.edges_to_sparse(cur_fragile, n)).nnz == 0
        if edges_are_same or np.isclose(max_obj_value, cur_obj_value):
            break
        else:
            prev_fragile = cur_fragile.copy()
            max_obj_value = cur_obj_value

    return cur_fragile, cur_obj_value


def relaxed_qclp(adj, alpha, fragile, local_budget, reward, teleport, global_budget=None, upper_bounds=None):
    """
    Solves the linear program associated with the relaxed QCLP.

    Parameters
    ----------
    adj : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    alpha : float
        (1-alpha) teleport[v] is the probability to teleport to node v.
    fragile : np.ndarray, shape [?, 2]
        Fragile edges that are under our control.
    local_budget : np.ndarray, shape [n]
        Maximum number of local flips per node.
    reward : np.ndarray, shape [n]
        Reward vector.
    teleport : np.ndarray, shape [n]
        Teleport vector.
    global_budget : int
        Global budget.
    upper_bounds : np.ndarray, shape [n]
        Upper bound for the values of x_i.
    Returns
    -------
    xval : np.ndarray, shape [n+len(fragile)]
        The value of the decision variables.
    opt_fragile : np.ndarray, shape [?, 2]
        Optimal fragile edges.
    obj_value : float
        Optimal objective value.
    """
    n = adj.shape[0]
    n_fragile = len(fragile)
    n_states = n + n_fragile

    adj = adj.copy()
    adj_clean = adj.copy()

    # turn off all existing edges before starting
    adj = adj.tolil()
    adj[fragile[:, 0], fragile[:, 1]] = 0

    # add an edge from the source node to the new auxiliary variables
    source_to_aux = sp.lil_matrix((n, n_fragile))
    source_to_aux[fragile[:, 0], np.arange(n_fragile)] = 1

    original_nodes = sp.hstack((adj,
                                source_to_aux
                                ))
    original_nodes = sp.diags(1 / original_nodes.sum(1).A1) @ original_nodes

    # transitions among the original nodes are discounted by alpha
    original_nodes[:, :n] *= alpha

    # add an edge from the auxiliary variables back to the source node
    aux_to_source = sp.lil_matrix((n_fragile, n))
    aux_to_source[np.arange(n_fragile), fragile[:, 0]] = 1
    turned_off = sp.hstack((aux_to_source,
                            sp.csr_matrix((n_fragile, n_fragile))))

    # add an edge from the auxiliary variables to the destination node
    aux_to_dest = sp.lil_matrix((n_fragile, n))
    aux_to_dest[np.arange(n_fragile), fragile[:, 1]] = 1
    turned_on = sp.hstack((aux_to_dest,
                           sp.csr_matrix((n_fragile, n_fragile))))
    # transitions from aux nodes when turned on are discounted by alpha
    turned_on *= alpha

    trans = sp.vstack((original_nodes, turned_off, turned_on)).tocsr()

    states = np.arange(n + n_fragile)
    states = np.concatenate((states, states[-n_fragile:]))

    c = np.zeros(len(states))
    # reward for the original nodes
    c[:n] = reward
    # negative reward if we are going back to the source node
    c[n:n + n_fragile] = -reward[fragile[:, 0]]

    one_hot = sp.eye(n_states).tocsr()
    A = one_hot[states] - trans

    b = np.zeros(n_states)
    b[:n] = (1 - alpha) * teleport

    x = cp.Variable(len(c), nonneg=True)

    # set up the sums of auxiliary variables for local and global budgets
    frag_adj = sp.lil_matrix((len(c), len(c)))

    # the indices of the turned off/on auxiliary nodes
    idxs_off = n + np.arange(n_fragile)
    idxs_on = n + n_fragile + np.arange(n_fragile)

    # if the edge exists in the clean graph use the turned off node, otherwise the turned on node
    exists = adj_clean[fragile[:, 0], fragile[:, 1]].A1
    idx_off_on_exists = np.where(exists, idxs_off, idxs_on)
    # each source node is matched with the correct auxiliary node (off or on)
    frag_adj[fragile[:, 0], idx_off_on_exists] = 1

    deg = (trans != 0).sum(1).A1
    unique = np.unique(fragile[:, 0])

    # the local budget constraints are sum_i ( x_i_{on/off} * deg_i ) <= budget * x_i)
    # we index only on the unique source nodes to avoid trivial constraints
    budget_constraints = [
        cp.multiply((frag_adj @ x)[unique], deg[unique]) <= cp.multiply(local_budget[unique], x[unique])]

    if global_budget is not None and upper_bounds is not None:
        # if we have a bounds matrix (for any PPR vector) we need to compute the upper bounds for the teleport
        if len(upper_bounds.shape) == 2:
            upper_bounds = teleport @ upper_bounds

        # do not consider upper_bounds that are zero
        nnz_unique = unique[upper_bounds[unique] != 0]
        # the global constraint is sum_i ( x_i_{on/off} * deg_i / upper(x_i) ) <= budget )
        global_constraint = [(frag_adj @ x)[nnz_unique] @ (deg[nnz_unique] / upper_bounds[nnz_unique]) <= global_budget]
    else:
        if global_budget is not None or upper_bounds is not None:
            warnings.warn('Either global_budget or upper_bounds is provided, but not both. '
                          'Solving using only local budget.')
        global_constraint = []

    print(x.shape, A.shape, b.shape)
    prob = cp.Problem(objective=cp.Maximize(c * x),
                      constraints=[x * A == b] + budget_constraints + global_constraint)
    prob.solve(solver='GUROBI', verbose=True)

    assert prob.status == 'optimal'

    xval = x.value
    # reshape the decision variables such that x_ij^0 and x_ij^1 are in the same row
    opt_fragile_on_off = xval[n:].reshape(2, -1).T.argmax(1)
    opt_fragile = fragile[opt_fragile_on_off != exists]

    obj_value = prob.value
    return xval, opt_fragile, obj_value, prob


def upper_bounds_max_ppr_target(adj, alpha, fragile, local_budget, target):
    """
    Computes the upper bound for x_target for any teleport vector.

    Parameters
    ----------
    adj : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    alpha : float
        (1-alpha) teleport[v] is the probability to teleport to node v.
    fragile : np.ndarray, shape [?, 2]
        Fragile edges that are under our control.
    local_budget : np.ndarray, shape [n]
        Maximum number of local flips per node.
    target : int
        Target node.

    Returns
    -------
    upper_bounds: np.ndarray, shape [n]
        Computed upper bounds.
    """
    n = adj.shape[0]
    z = np.zeros(n)
    z[target] = 1
    opt_fragile, _ = policy_iteration(adj=adj, alpha=alpha, fragile=fragile, local_budget=local_budget,
                                      reward=z, teleport=z)
    adj_flipped = flip_edges(adj, opt_fragile)

    # gets one column from the PPR matrix
    # corresponds to the PageRank score value of target for any teleport vector (any row)
    pre_inv = sp.eye(n) - alpha * sp.diags(1 / adj_flipped.sum(1).A1) @ adj_flipped
    ppr = (1 - alpha) * sp.linalg.gmres(pre_inv, z)[0]

    correction = correction_term(adj, opt_fragile, fragile)

    upper_bounds = ppr / correction
    return upper_bounds


def upper_bounds_max_ppr_all_nodes(adj, alpha, fragile, local_budget, do_parallel=True):
    """
    Computes the upper bounds needed for QCLP for all nodes at once.

    Parameters
    ----------
    adj : sp.spmatrix, shape [n, n]
        Sparse adjacency matrix.
    alpha : float
        (1-alpha) teleport[v] is the probability to teleport to node v.
    fragile : np.ndarray, shape [?, 2]
        Fragile edges that are under our control.
    local_budget : np.ndarray, shape [n]
        Maximum number of local flips per node.
    do_parallel : bool
        Parallel

    Returns
    -------
    upper_bounds: np.ndarray, shape [n]
        Computed upper bounds.

    """

    n = adj.shape[0]

    if do_parallel:
        parallel = Parallel(20)
        results = parallel(delayed(upper_bounds_max_ppr_target)(adj, alpha, fragile, local_budget, target)
                           for target in range(n))
        upper_bounds = np.column_stack(results)
    else:
        upper_bounds = np.zeros((n, n))
        for target in trange:
            upper_bounds[:, target] = upper_bounds_max_ppr_target(
                adj=adj, alpha=alpha, fragile=fragile, local_budget=local_budget, target=target)

    return upper_bounds


def worst_margin_local(adj, alpha, fragile, budget, logits, true_class, other_class):
    # min(true - other_class) = -max(other_class-true_class)
    # therefore multiply the final result by -1
    reward = logits[:, other_class] - logits[:, true_class]

    # pick any teleport vector
    teleport = np.zeros_like(reward)
    teleport[0] = 1

    opt_fragile, obj_value = policy_iteration(
        adj=adj, alpha=alpha, fragile=fragile, local_budget=budget, reward=reward, teleport=teleport)

    adj_flipped = flip_edges(adj, opt_fragile)
    ppr_flipped = propagation_matrix(adj=adj_flipped, alpha=alpha)

    return true_class, other_class, -obj_value, ppr_flipped, opt_fragile


def k_squared_parallel(adj, alpha, fragile, local_budget, logits):
    parallel = Parallel(20)

    n, nc = logits.shape

    # any teleport vector can be used
    teleport = np.zeros(n)
    teleport[0] = 1

    results = parallel(delayed(worst_margin_local)(
        adj, alpha, fragile, local_budget, logits, c1, c2)
                       for c1 in range(nc)
                       for c2 in range(nc)
                       if c1 != c2)

    results_dict = {}
    for c1, c2, _, ppr_flipped, _ in results:
        results_dict[(c1, c2)] = {'ppr': ppr_flipped}

    return results_dict


def worst_margins_given_k_squared(k_squared, labels, logits):
    n, nc = logits.shape
    worst_margins_all = np.ones((nc, nc, n)) * np.inf
    worst_pprs_all = np.zeros((nc, nc, n, n))

    for c1 in range(nc):
        for c2 in range(nc):
            if c1 != c2:
                worst_pprs_all[c1, c2] = k_squared[c1, c2]['ppr']
                worst_margins_all[c1, c2] = (k_squared[c1, c2]['ppr'] @ (logits[:, c1] - logits[:, c2]))

    worst_margins = np.nanmin(worst_margins_all[labels, :, np.arange(n)], 1)
    # worst_classes = np.nanargmin(worst_margins_all[labels, :, np.arange(n)], 1)
    # worst_pprs = worst_pprs_all[labels, worst_classes, np.arange(n)]

    return worst_margins, worst_pprs_all, worst_margins_all
