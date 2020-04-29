#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

"""
==============================================================
Hierarchical clustering (:mod:`pyannote.core.utils.hierarchy`)
==============================================================

"""

from typing import Text, Callable, List, Tuple
import numpy as np
import scipy.cluster.hierarchy
from .distance import to_condensed
from .distance import to_squared
from .distance import l2_normalize
from .distance import pdist
from .distance import cdist


def linkage(X, method='single', metric='euclidean',
            cannot_link: List[Tuple[int, int]] = None,
            must_link: List[Tuple[int, int]] = None):
    """Same as scipy.cluster.hierarchy.linkage with more metrics and methods
    `cannot_link` and `must_link` are only supported with 'pool' method
    """

    if method == 'pool':
        return pool(X, metric=metric, pooling_func=None,
                    cannot_link=cannot_link, must_link=must_link)
    elif cannot_link or must_link:
        msg =(f'cannot_link and must_link are only supported with "pool" method, '
              f'got {method} instead.')
        raise ValueError(msg)

    # corner case when using non-euclidean distances with methods
    # designed for the euclidean distance
    if metric != 'euclidean' and method in ['centroid', 'median', 'ward']:
        # Those 3 methods only work with 'euclidean' distance.
        # Therefore, one has to unit-normalized embeddings before
        # comparison in case they were optimized for 'cosine' (or 'angular')
        # distance.
        X = l2_normalize(X)
        metric = 'euclidean'

    distance = pdist(X, metric=metric)
    return scipy.cluster.hierarchy.linkage(distance, method=method,
                                           metric=metric)


def pool(
    X: np.ndarray,
    metric: Text = "euclidean",
    pooling_func: Callable = None,
    cannot_link: List[Tuple[int, int]] = None,
    must_link: List[Tuple[int, int]] = None,
):
    """'pool' linkage

    Parameters
    ----------
    X : np.ndarray
        (n_samples, dimension) obversations.
    metric : {"euclidean", "cosine", "angular"}, optional
        Distance metric. Defaults to "euclidean"
    pooling_func: callable

    cannot_link : list of pairs
        Pairs of indices of observations that cannot be linked. For instance,
        [(1, 2), (5, 6)] means that first and second observations cannot end up
        in the same cluster, as well as 5th and 6th obversations.
    """

    if pooling_func is None:

        def pooling_func(
            C_u: np.ndarray, C_v: np.ndarray, X_u: np.ndarray, X_v: np.ndarray
        ) -> np.ndarray:
            """
            Parameters
            ----------
            C_u : (dimension, ) np.ndarray
                Centroid of u^th cluster.
            C_v : (dimension, ) np.ndarray
                Centroid of v^th cluster.
            X_u : (n_samples_u, dimension) np.ndarray
                Elements of u^th cluster.
            X_v : (n_samples_v, dimension) np.ndarray
                Elements of v^th cluster.

            Returns
            -------
            C : (dimension, ) np.ndarray
                Centroid of newly formed cluster.
            """
            S_u = len(X_u)
            S_v = len(X_v)
            return (C_u * S_u + C_v * S_v) / (S_u + S_v)

    # obtain number of original observations
    n, dimension = X.shape

    # K[j] contains the index of the cluster to which
    # the jth observation is currently assigned
    K = np.arange(n)

    # S[k] contains the current size of kth cluster
    S = np.zeros(2 * n - 1, dtype=np.int16)
    S[:n] = 1

    # C[k] contains the centroid of kth cluster
    C = np.zeros((2 * n - 1, dimension))
    # at the beginning, each observation is assigned to its own cluster
    C[:n, :] = X

    # clustering tree (aka dendrogram)
    # Z[i, 0] and Z[i, 1] are merged at ith iteration
    # Z[i, 2] is the distance between Z[i, 0] and Z[i, 1]
    # Z[i, 3] is the total number of original observation in the newly formed cluster
    Z = np.zeros((n - 1, 4))

    # convert condensed pdist matrix for the `n` original observation to a
    # condensed pdist matrix for the `2n-1` clusters (including the `n`
    # original observations) that will exist at some point during the process.
    D = np.infty * np.ones((2 * n - 1) * (2 * n - 2) // 2)
    D[to_condensed(2 * n - 1, *to_squared(n, np.arange(n * (n - 1) // 2)))] = pdist(
        X, metric=metric
    )

    def merge(u, v, iteration, constraint=False):
        """Merge two clusters

        Parameters
        ----------
        u, v : int
            Indices of clusters to merge.
        iteration : int
            Current clustering iteration.
        constraint : bool
            Set to True to indicate that this merge is coming from a 'must_link'
            constraint. This will artificially set Z[iteration, 2] to 0.0.

        Returns
        -------
        uv : int
            Indices of resulting cluster.

        Raises
        ------
        "ValueError" in case of conflict between "must_link" and "cannot_link"
        constraints.

        """

        k = to_condensed(2 * n - 1, u, v)

        if constraint and D[k] == np.infty:
            w = u if u < n else v
            msg = f"Found a conflict between 'must_link' and 'cannot_link' constraints for observation {w}."
            raise ValueError(msg)

        # keep track of ...
        # ... which clusters are merged at this iteration
        Z[iteration, 0] = v if S[v] > S[u] else v
        Z[iteration, 1] = u if Z[iteration, 0] == v else v

        # ... their distance
        Z[iteration, 2] = 0.0 if constraint else D[k]

        # ... the size of the newly formed cluster
        Z[iteration, 3] = S[u] + S[v]
        S[n + iteration] = S[u] + S[v]

        # merged clusters are now empty...
        S[u] = 0
        S[v] = 0

        # compute centroid of newly formed cluster
        C[n + iteration] = pooling_func(C[u], C[v], X[K == u], X[K == v])

        # move observations of merged clusters into the newly formed cluster
        K[K == u] = n + iteration
        K[K == v] = n + iteration

        # compute distance to newly formed cluster
        # (only for clusters that still exists, i.e. those that are not empty)
        empty = S[: n + iteration] == 0
        k = to_condensed(2 * n - 1, n + iteration, np.arange(n + iteration)[~empty])
        D[k] = cdist(
            C[np.newaxis, n + iteration, :],
            C[: n + iteration, :][~empty, :],
            metric=metric,
        )

        # condensed indices of all (u, _) and (v, _) pairs
        _u = to_condensed(2 * n - 1, u, np.arange(u))
        u_ = to_condensed(2 * n - 1, u, np.arange(u + 1, n + iteration))
        _v = to_condensed(2 * n - 1, v, np.arange(v))
        v_ = to_condensed(2 * n - 1, v, np.arange(v + 1, n + iteration))

        # propagate "cannot link" constraints to newly formed cluster
        if cannot_link:
            x, _ = to_squared(2 * n - 1, _u[D[_u] == np.infty])
            D[to_condensed(2 * n - 1, n + iteration, x)] = np.infty
            _, x = to_squared(2 * n - 1, u_[D[u_] == np.infty])
            D[to_condensed(2 * n - 1, n + iteration, x)] = np.infty
            x, _ = to_squared(2 * n - 1, _v[D[_v] == np.infty])
            D[to_condensed(2 * n - 1, n + iteration, x)] = np.infty
            _, x = to_squared(2 * n - 1, v_[D[v_] == np.infty])
            D[to_condensed(2 * n - 1, n + iteration, x)] = np.infty

        # distance to merged clusters u and v no longer exist
        D[_u] = np.infty
        D[u_] = np.infty
        D[_v] = np.infty
        D[v_] = np.infty

        k = to_condensed(2 * n - 1, n + iteration, np.arange(n + iteration)[empty])
        D[k] = np.infty

        return n + iteration

    iteration = 0

    # take "cannot link" constraints into account by artifically setting the
    # distance between corresponding observations to infinity.
    if cannot_link is not None:
        u, v = zip(*cannot_link)
        D[to_condensed(2 * n - 1, u, v)] = np.infty

    # take "must link" constraints into account by merging corresponding
    # observations regardless of their actual similarity. this might lead to
    # weird clustering results when merged observations are very dissimilar.

    if must_link is not None:
        # find connected components in "must link" graph
        graph = np.zeros((n, n), dtype=np.int8)
        for u, v in must_link:
            graph[u, v] = 1
        _, K_init = connected_components(
            csr_matrix(graph), directed=False, return_labels=True
        )

        # merge observations within each connected components
        for k, count in Counter(K_init).items():
            if count < 2:
                continue
            u, *others = np.where(K_init == k)[0]
            for v in others:
                u = merge(u, v, iteration, constraint=True)
                iteration += 1

    # iterate until one cluster remains
    for iteration in range(iteration, n - 1):

        # find two most similar clusters
        k = np.argmin(D)

        # when cannot_link constraints prevents any further merging,
        # choose an arbitrary (u, v) pair among remaining clusters
        if D[k] == np.infty:
            u, v, *_ = np.where(S > 0)[0]
        else:
            u, v = to_squared(2 * n - 1, k)

        _ = merge(u, v, iteration)

    return Z


def fcluster_auto(X, Z, metric='euclidean'):
    """Forms flat clusters using within-class sum of square elbow criterion

    Parameters
    ----------
    X : `np.ndarray`
        (n_samples, n_dimensions) feature vectors.
    Z : `np.ndarray`
        The hierarchical clustering encoded with the matrix returned by the
        `linkage` function.
    metric : `str`
        The distance metric to use. See `pdist` function for a list of valid
        distance metrics.

    Returns
    -------
    T : ndarray
        An array of length n. T[i] is the flat cluster number to which
        original observation i belongs.

    Reference
    ---------
    H. Delgado, X. Anguerra, C. Fredouille, J. Serrano. "Fast Single- and
    Cross-Show Speaker Diarization Using Binary Key Speaker Modeling".
    IEEE Transactions on Audio Speech and Language Processing
    """

    # within-class sum of squares
    wcss = []
    for threshold in Z[:, 2]:
        y_t = scipy.cluster.hierarchy.fcluster(Z, threshold,
                                               criterion='distance')
        D = []
        for k in np.unique(y_t):
            Xk = X[y_t == k]
            Ck = np.mean(Xk, axis=0, keepdims=True)
            D.append(cdist(Ck, Xk, metric=metric).reshape(-1, ))
        wcss.append(np.mean(np.hstack(D)**2))
    wcss = np.array(wcss)

    # elbow criterion
    n = len(X)

    # after first step, there n-1 clusters (x1)
    x1, y1 = n - 1, wcss[0]
    # after last step, there is 1 cluster (x2)
    x2, y2 = 1, wcss[-1]

    # equation of line passing at both points
    # ax + by + c = 0
    a = (y2 - y1) / (x2 - x1)
    b = 1
    c = (x2 * y1 - x1 * y2) / (x1 - x2)

    # elbow is at maximum distance to this line
    distance = np.abs(a * np.arange(1, n) + b * wcss + c) / np.sqrt(a**2 + b**2)
    threshold = Z[np.argmax(distance), 2]

    return scipy.cluster.hierarchy.fcluster(Z, threshold, criterion='distance')
