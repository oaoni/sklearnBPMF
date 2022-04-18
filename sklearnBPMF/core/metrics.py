import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.linalg import svd
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.spatial.distance import cdist

def corr_metric(predicted, measured):
    corr = np.corrcoef(predicted, measured)

    return corr[0][-1]

def distance_metric(predicted, measured, metric, param):
    distance = cdist(np.array(predicted).reshape(1,-1),
                     np.array(measured).reshape(1,-1),
                     metric,
                     param)

    return distance[0][0]

def effective_rank(m):
    """
    effective rank of matrix = entropy of singular value distribution
    one heuristic for choosing number of components to keep
    """

    u, s, vt = svd(m)
    p = s/s.sum()
    return np.exp(entropy(p))

def rank_k_leverage_scores(m, k, kernel=None, **kwargs):
    """
    row leverage scores of a matrix
    truncate svd to rank k, take left singular vectors, square, normalize by sum
    kernel is present in case you want to do kernelized version = svd on pairwise distance matrix
    """

    if kernel is None:
        u, s, vt = svd(m)
    else:
        K = pairwise_kernels(m, metric=kernel, **kwargs)
        u, s, vt = svd(K)
    lev_scores = (u[:, :k]**2).sum(axis=1)
    if isinstance(m, pd.DataFrame):
        lev_scores = pd.Series(lev_scores, index=m.index)
    return lev_scores/lev_scores.sum()

def sampling_distribution(m, k, kernel=None, **kwargs):
    """
    def produce a sampling distribution for a *symmetric* matrix
    """

    lev_scores = rank_k_leverage_scores(m, k, kernel, **kwargs)
    lev_sampling = pd.DataFrame(np.outer(lev_scores, lev_scores),
                                index=lev_scores.index, columns=lev_scores.index)
    return lev_scores, lev_sampling

def leverage_update(m, k, n=1, kernel=None, **kwargs):
    """
    predict which entry to read next based on leverage scores
    assumes matrix is symmetric, computes row leverage scores to produce a probability distrbituion on rows,
    and then produces a sampling distribution over the matrix by taking the outer product
    parameters: k = rank to reduce to when computing leverage scores
    (can potentially use effective_rank above or say effective_rank/2 as heuristics)
    n = number of indices to return
    returns flat indices that are always in the upper triangle of the matrix
    """

    lev_scores, lev_sampling = sampling_distribution(m, k, kernel=None, **kwargs)
    # take only entries in upper triangle
    upper_tri_ind = np.triu_indices(lev_sampling.shape[0], k=1)
    #
    p = lev_sampling.values[upper_tri_ind]
    p = p/p.sum()
    flat_ind = np.ravel_multi_index(upper_tri_ind, lev_sampling.shape)
    if n == 1:
        return np.random.choice(flat_ind,n,replace=False, p=p)[0]
    else:
        return np.random.choice(flat_ind,n,replace=False, p=p)

def compute_leverage(M):

    target = M.copy()
    # lev_sampling is matrix of leverage scores for each entry
    # below uses a (poorly founded) heuristic for choosing rank to truncate svd to
    lev_scores, lev_sampling = sampling_distribution(target, int(effective_rank(target)/2))

    return lev_scores, lev_sampling
