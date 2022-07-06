import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.linalg import svd
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.spatial.distance import cdist
from sklearnBPMF.data.utils import verify_pdframe

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

def item_rank(M):
    '''Order user-item pairs based of predicted item ratings for each user'''

    item_dict = {}
    rating_dict = {}
    for user,row in M.iterrows():
        # Absolute value of ratings/predictions used for strong negative and positive values.
        sorted_row = row.abs().sort_values(ascending=False).index
        item_ = row[sorted_row].index.values
        rating_ = row[sorted_row].values

        item_dict[user] = item_
        rating_dict[user] = rating_

    item_df = pd.DataFrame(item_dict)
    rating_df = pd.DataFrame(rating_dict)

    return item_df, rating_df

# score rank
def relevance_score(pred_rank, true_M):

    rel_score = pd.DataFrame(pred_rank.melt()\
                .apply(lambda x: true_M.loc[x['variable'],x['value']],axis=1).values\
                .reshape(pred_rank.shape[0],-1, order='F'),columns = pred_rank.columns)

    return rel_score

# Mean reciprocal rank (Average recriprocal hit ratio)
def reciprocal_rank(pred_M, true_M, k):
    """ Mean Reciprocal Rank in top k items
        pred_M: predicted rating matrix
        true_M: true rating matrix
    """

    pred_rank,_ = item_rank(pred_M)
    top_rank = pred_rank.iloc[:k,:]

    top_rel = relevance_score(top_rank, true_M)

    rank_mult = 1/np.arange(1,k+1)
    reciprocal_rel = top_rel.abs() * rank_mult.reshape(-1,1)

    hit_ratio = reciprocal_rel.sum(axis=0)
    mean_hit_ratio = hit_ratio.mean()

    return mean_hit_ratio, hit_ratio

# Mean average precision
def average_precision(pred_M,true_M,k,cutoff=0.75):
    """ Mean average precision @ k
        pred_M: predicted rating matrix
        true_M: true rating matrix
    """

    pred_rank,_ = item_rank(pred_M)
    top_rank = pred_rank.iloc[:k,:]

    top_rel = relevance_score(top_rank, true_M)

    rel_quantile = true_M.abs().quantile(q=cutoff,axis=0)

    precision = (top_rel.abs() >= rel_quantile).sum(axis=0)/k

    mean_precision = precision.mean()

    return mean_precision, precision

# Mean average recall
def average_recall(pred_M,true_M,k,cutoff=0.75):
    """ Mean average recall @ k
        pred_M: predicted rating matrix
        true_M: true rating matrix
    """

    pred_rank,_ = item_rank(pred_M)
    top_rank = pred_rank.iloc[:k,:]

    top_rel = relevance_score(top_rank, true_M)

    rel_quantile = true_M.abs().quantile(q=cutoff,axis=0)
    rel_docs = sum(true_M.iloc[:,0].abs() >= rel_quantile[0])

    recall = (top_rel.abs() >= rel_quantile).sum(axis=0)/rel_docs

    mean_recall = recall.mean()

    return mean_recall, recall

# Discounted cumulative gain
def discounted_gain(pred_M,true_M,k):
    """ Discounted cumulative gain in top k items
        pred_M: predicted rating matrix
        true_M: true rating matrix
    """

    pred_rank,_ = item_rank(pred_M)
    top_rank = pred_rank.iloc[:k,:]

    top_rel = relevance_score(top_rank, true_M)

    rank_discount = 1/np.log2(np.arange(2,k+2))
    discounted_gain = top_rel.abs() * rank_discount.reshape(-1,1)

    cumulative_gain = discounted_gain.sum(axis=0)
    mean_dcg = cumulative_gain.mean()

    return mean_dcg, cumulative_gain

# Normalized discounted cumulative gain
def normalized_gain(pred_M,true_M,k):
    """ Normalized discounted cumulative gain in top k items
        pred_M: predicted rating matrix
        true_M: true rating matrix
    """

    mean_dcg, cumulative_gain = discounted_gain(pred_M,true_M,k)

    true_rank, true_rel = item_rank(true_M)
    top_rank = true_rank.iloc[:k,:]
    top_rel = true_rel.iloc[:k,:]

    rank_discount = 1/np.log2(np.arange(2,k+2))

    ideal_discounted_gain = top_rel.abs() * rank_discount.reshape(-1,1)

    ideal_cumulative_gain = ideal_discounted_gain.sum(axis=0)

    normalized_cg = cumulative_gain / ideal_cumulative_gain
    mean_ndcg = mean_dcg / ideal_cumulative_gain.mean()

    return mean_ndcg, normalized_cg

def score_rank(X,Xhat,k):
    """Compute @k ranking metrics"""

    reciprocal_r = reciprocal_rank(Xhat, X, k)[0]
    mean_precision = average_precision(Xhat, X, k)[0]
    mean_recall = average_recall(Xhat, X, k)[0]
    ndcg = normalized_gain(Xhat, X, k)[0]

    return reciprocal_r, mean_precision, mean_recall, ndcg

def score_completion(X, X_pred, S_test, name, k_metrics=False, k=20):
    ''''Produce scoring metrics (rmse, corr(pearson,), frobenius, relative error)'''

    X, X_pred, S_test = verify_pdframe(X, X_pred, S_test)

    bool_mask = S_test == 1
    Xhat = X_pred.set_axis(X.index,axis=0).set_axis(X.index,axis=1)

    predicted = Xhat[bool_mask].stack()
    measured = X[bool_mask].stack()

    error = predicted - measured

    frob = np.linalg.norm(Xhat - X, 'fro')
    rel_frob = frob/(np.linalg.norm(X, 'fro'))
    rmse = np.sqrt(np.mean((error**2)))
    rel_rmse = rmse/(np.sqrt(np.mean((measured**2))))
    spearman = predicted.corr(measured, method='spearman')
    pearson = predicted.corr(measured, method='pearson')

    metric_dict = {'{}_frob'.format(name):frob, '{}_rel_frob'.format(name):rel_frob, '{}_rmse'.format(name):rmse,
    '{}_rel_rmse'.format(name):rel_rmse, '{}_spearman'.format(name):spearman, '{}_pearson'.format(name):pearson}

    if k_metrics:

        reciprocal_r, mean_precision, mean_recall, ndcg = score_rank(X,Xhat,k)

        metric_dict['{}_reciprocal_r'.format(name)] = reciprocal_r
        metric_dict['{}_mean_precision'.format(name)] = mean_precision
        metric_dict['{}_mean_recall'.format(name)] = mean_recall
        metric_dict['{}_ndcg'.format(name)] = ndcg

    return metric_dict
