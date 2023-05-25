import logging
from typing import List

import numpy as np
import scipy

### This file cotains code from https://github.com/yanaiela/amnesic_probing used to generate projections which randomly remove directions


def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """

    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T)  # orthogonal basis

    w_basis = w_basis * np.sign(w_basis[0][0])  # handle sign ambiguity !!! Old line had no effect: w_basis * np.sign(w_basis[0][0])
    P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace

    return P_W


def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: List[np.ndarray], input_dim: int):
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
    :param input_dim: input dim
    """

    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis=0)
    P = I - get_rowspace_projection(Q)

    return P


def debias_by_specific_directions(directions: List[np.ndarray], input_dim: int):
    """
    the goal of this function is to perform INLP on a set of user-provided directions
    (instead of learning those directions).
    :param directions: List of vectors, as numpy arrays.
    :param input_dim: dimensionality of the vectors.
    """

    rowspace_projections = []

    for v in directions:
        P_v = get_rowspace_projection(v)
        rowspace_projections.append(P_v)

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P


def create_rand_dir_projection(dim, n_coord):
    # creating random directions (vectors) within the range of -0.5 : 0.5
    rand_directions = [np.random.rand(1, dim) - 0.5 for _ in range(n_coord)]

    # finding the null-space of random directions
    rand_direction_projection = debias_by_specific_directions(rand_directions, dim)
    return rand_direction_projection


def create_rand_dir_from_orth_basis_projection(X, n_coord):
    # get orth basis of X
    orth_basis = scipy.linalg.orth(X.T)
    orth_basis_rank = np.linalg.matrix_rank(orth_basis)

    if n_coord > orth_basis_rank:
        logging.info(
            f"Subspace rank to be removed larger than rank of the space ({n_coord} > {orth_basis_rank}). Setting rank to {orth_basis_rank}"
        )
        n_coord = orth_basis_rank

    # get n_coord random indices from possible directions of orth basis
    rand_indices = np.random.choice(orth_basis_rank, n_coord, replace=False)
    projections = [orth_basis[:, i].reshape(-1, 1).dot(orth_basis[:, i].reshape(-1, 1).T) for i in rand_indices]

    final_P = np.eye(X.shape[1]) - np.sum(projections, axis=0)

    # finding the null-space of the directions
    # rand_direction_projection = debias_by_specific_directions(rand_directions, X.shape[1])
    return final_P
