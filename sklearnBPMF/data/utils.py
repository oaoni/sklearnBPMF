import pandas as pd
import numpy as np
import umap.umap_ as umap
from sklearn.cluster import KMeans
import networkx as nx
from numpy import linalg as npla
import random
import scipy
import smurff
from scipy.sparse import coo_matrix
from scipy.cluster.hierarchy import linkage, dendrogram

import os
import tempfile
import scipy.io as sio
from hashlib import sha256

try:
    import urllib.request as urllib_request  # for Python 3
except ImportError:
    import urllib2 as urllib_request  # for Python 2

urls = {
        "chembl-IC50-346targets.mm" :
        (
            "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm",
            "10c3e1f989a7a415a585a175ed59eeaa33eff66272d47580374f26342cddaa88",
        ),

        "chembl-IC50-compound-feat.mm" :
        (
            "http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compound-feat.mm",
            "f9fe0d296272ef26872409be6991200dbf4884b0cf6c96af8892abfd2b55e3bc",
        ),
        "pkis2-compound-target-feat.xlsx":
        (
            "https://doi.org/10.1371/journal.pone.0181585.s004",
            "48ead22a1f860cd0d5096fa87d5acd329f722fe8d65e693bb0be682a333e2a2c"
        )
}

def load_gi_example(frac, data_path, side_path, cluster=None):
    """
    loads gi delta data
    """
    path_dataset = data_path
    side_dataset = side_path

    M = pd.read_csv(path_dataset,index_col=0)
    side = pd.read_csv(side_dataset,index_col=0)

    if cluster:
        linkage_ = linkage(M, method=cluster) #average, ward, single, complete, centroid, weighted, median
        dendrogram_ = dendrogram(linkage_, no_plot=True)
        clust_index = dendrogram_['leaves']
        M = M.iloc[clust_index,clust_index]
        side = side.iloc[clust_index,:]

    M,S_train,S_test,M_train,M_test = makeTrainTest(M,frac,use_upper=True)

    return [M,S_train,S_test,M_train,M_test],side

def makeTrainTest(M,frac=None,n_frac=None,use_upper=True,use_index=False,avoid_diag=False,weights=None,diag_for_trainset=False):

    #Two functions for sampling from the phenotype matrix
    random.seed(30)

    #Sample a percentage of the genes
    M_train, M_test, S_train, S_test = sample_mask(M,frac,n_frac,use_upper=use_upper,use_index=use_index,
                                                    avoid_diag=avoid_diag,weights=weights,
                                                    diag_for_trainset=diag_for_trainset)
    M_train = M_train.values
    M_test = M_test.values
    S_train = S_train.values
    S_test = S_test.values

    return M,S_train,S_test,M_train,M_test

def saveDataFrameH5(data_dict, fname='matrix_data.h5', verbose=True):

    # Check if fname exists already, and raise error
    if os.path.isfile(fname):
        raise FileExistsError

    s = pd.HDFStore(fname)

    for key, value in data_dict.items():
        try:
            s[key] = value
        except TypeError:
            s[key] = pd.DataFrame(value)

    if verbose:
        print('Pandas h5py file saved to {}'.format(fname))

    s.close()


def saveTrainTestH5(data_dict, fname='matrix_data.h5'):
    print("Function deprecated, replaced by saveDataFrameH5")
    s = pd.HDFStore(fname) # Ideally, check if exists first

    for key, value in data_dict.items():
        try:
            s[key] = value
        except TypeError:
            s[key] = pd.DataFrame(value)

    print('Pandas h5py file saved to {}'.format(fname))
    s.close()

def loadDataFrameH5(fname):
    data_dict = {}

    # Check if fname exists
    if not os.path.isfile(fname):
        raise FileNotFoundError

    with pd.HDFStore(fname) as s:
        h5data = [x.split('/')[1] for x in s.keys()]

        for key in h5data:
            data_dict[key] = s.get(key)

    return data_dict

def loadTrainTestH5(fname):
    print("Function deprecated, replaced by loadDataFrameH5")
    data_dict = {}

    s = pd.HDFStore(fname)
    h5data = [x.split('/')[1] for x in s.keys()]

    for key in h5data:
        data_dict[key] = s.get(key)

    s.close()
    return data_dict

def upper_triangle(M, k=1):
    """Return the upper triangular part of a matrix in stacked format (i.e. as a vector)
    """
    keep = np.triu(np.ones(M.shape), k=k).astype('bool').reshape(M.size)
    # have to use dropna=FALSE on stack, otherwise will secretly drop nans and upper triangle
    # will not behave as expected!!
    return M.stack(dropna=False).loc[keep]

def sample_mask(M,frac,n_frac,use_upper=False,use_index=False,
                random_state=np.random.RandomState(30),
                avoid_diag=False,weights=None,
                diag_for_trainset=False):
    """ Samples a fraction of marix elements # Seperate matrices for sampling and selecting data with weights
    returns: train/test matrix (M), and Mask (S)
    """

    if isinstance(n_frac,int):
        sample_kwargs = {'n':n_frac}
    elif isinstance(frac,float):
        sample_kwargs = {'frac':frac}

    if use_upper:
        # k = 1 if avoid_diag else 0
        k = 0
        weights = weights.stack(dropna=False) if isinstance(weights,pd.DataFrame) else None

        #Samples from upper triangular. For symmetrical relational matrices.
        multInd = upper_triangle(M, k=k).sample(**sample_kwargs, replace=False,
                                                random_state=random_state,
                                                weights=weights).index

        #Initialize indicators
        S_train = pd.DataFrame(0, index=M.index, columns=M.columns)
        S_test = pd.DataFrame(1, index=M.index, columns=M.columns)

        if avoid_diag:
            np.fill_diagonal(S_test.values, 0)

        if diag_for_trainset:
            np.fill_diagonal(S_train.values, 1)

        for Ind in multInd:
            S_train.loc[Ind] = 1
            S_test.loc[Ind] = 0

            S_train.loc[Ind[::-1]] = 1
            S_test.loc[Ind[::-1]] = 0

        M_train = M*S_train
        M_test = M*S_test

    elif use_index:
        #Samples from entire matrix
        #Produce multiindex of sampled elements (might want to return)

        index = weights if isinstance(weights,pd.Series) else None

        n_frac = int(((M.shape[0]**2)/2)*frac)
        ind = index.sample(**sample_kwargs,replace=True,weights=weights,
                          random_state=random_state).index

        multInd = pd.MultiIndex.from_arrays([ind[::2], ind[1::2]])

        #Initialize indicators
        S_train = pd.DataFrame(0, index=M.index, columns=M.columns)
        S_test = pd.DataFrame(1, index=M.index, columns=M.columns)

        if avoid_diag:
            np.fill_diagonal(S_test.values, 0)

        if diag_for_trainset:
            np.fill_diagonal(S_train.values, 1)

        for Ind in multInd:
            S_train.loc[Ind] = 1
            S_test.loc[Ind] = 0

            S_train.loc[Ind[::-1]] = 1
            S_test.loc[Ind[::-1]] = 0

        M_train = M*S_train
        M_test = M*S_test
    else:
        test_frac = 1-frac
        m,n = M.shape
        S = np.random.rand(m,n)
        S_train = (S < frac)
        S_test = (S >= (1-test_frac))
        M_train = np.multiply(M,S_train)
        M_test = np.multiply(M,S_test)

    return M_train, M_test, S_train, S_test

def side_process(side, form, near_n=15,min_d=0.1,plot=True):
    """ Various forms of processing side information for gi side information
    """

    if form == None:
        final_side = side

    elif form == 'sqrt_corr':
        final_side = scipy.linalg.sqrtm(side.T.corr())

    elif form == 'sqrt_only':
        final_side = scipy.linalg.sqrtm(side)

    else:
        if form == 'corr':
            processed_side = scipy.linalg.sqrtm(side.T.corr())

        elif form == 'raw_sim_set':
            side = side.values
            processed_side = umap_graph(side,near_n,min_d,plot=plot)
        elif form == 'corr_sim_set':
            side = side.T.corr().values
            processed_side = umap_graph(side,near_n,min_d,plot=plot)
        elif form == 'sqrt_sim_set':
            side = scipy.linalg.sqrtm(side.T.corr())
            processed_side = umap_graph(side,near_n,min_d,plot=plot)

        laplacian = graph_laplacian(processed_side)
        final_side = eigen(laplacian)

    return final_side

def umap_graph(data,near_n,min_d,plot=True):
    rand_s = np.random.RandomState(30)

    sim_set = umap.fuzzy_simplicial_set(data,near_n,random_state=rand_s,\
                                              metric="euclidean")
    sim_set = sim_set[0].toarray()
    if plot:

        transformer = umap.UMAP(n_neighbors=near_n,random_state=rand_s,min_dist=min_d).fit(data)
        raw_emb = transformer.transform(data);
        kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=rand_s)
        labels = kmeans.fit(raw_emb)

        node_l = labels.labels_

        nx.draw(nx.DiGraph(sim_set), arrows=False, node_size=30, width=0.1, node_color = node_l, cmap='viridis')

    return sim_set

def eigen(A):
    eigenValues, eigenVectors = npla.eigh(A)
    idx = np.argsort(eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    return (eigenValues, eigenVectors)

def graph_laplacian(W):
    # returns graph laplacian
    W = W - np.diag(np.diag(W))
    D = np.diagflat(np.sum(W, 1))
    L = D - W

    return L

def load_one(filename):
    (url, expected_sha) =  urls[filename]

    with tempfile.TemporaryDirectory() as tmpdirname:
            output = os.path.join(tmpdirname, filename)
            urllib_request.urlretrieve(url, output)
            actual_sha = sha256(open(output, "rb").read()).hexdigest()
            assert actual_sha == expected_sha
            matrix = sio.mmread(output)

    return matrix

def load_xlsx(filename):
    (url, expected_sha) =  urls[filename]

    with tempfile.TemporaryDirectory() as tmpdirname:
            output = os.path.join(tmpdirname, filename)
            urllib_request.urlretrieve(url, output)
            actual_sha = sha256(open(output, "rb").read()).hexdigest()
            assert actual_sha == expected_sha
            matrix = pd.read_excel(output, sheet_name=None,)

    return matrix

def load_chembl():
    """Downloads a small subset of the ChEMBL dataset.
    Returns
    -------
    ic50_train: sparse matrix
        sparse train matrix
    ic50_test: sparse matrix
        sparse test matrix
    side: sparse matrix
        sparse row features
    """

    # load bioactivity and features
    ic50 = load_one("chembl-IC50-346targets.mm")
    side = load_one("chembl-IC50-compound-feat.mm")

    return (ic50, side)

def load_pkis2():
    """Downloads a small subset of the PKIS2 dataset.
    Returns
    -------
    """

    # load bioactivity and features
    pkis = load_xlsx("pkis2-compound-target-feat.xlsx")

    return pkis

def smurffNormalizeData(M, M_train, M_test):

    train_centered, global_mean, _ = smurff.center_and_scale(coo_matrix(M_train),
                                          "global", with_mean = True, with_std = False)

    test_centered = coo_matrix(M_test)
    test_centered.data -= global_mean # only touch non-zeros
    all_centered = coo_matrix(M)
    all_centered.data -= global_mean # only touch non-zeros

    return train_centered, test_centered, all_centered

def to_sparse(data, inds, shape):

    row, col = list(zip(*inds))
    data_sparse = coo_matrix((data, (row, col)), shape=shape)

    return data_sparse

def quantile_mask(M, q, weights=None):

    if isinstance(weights, type(None)):
        Mat = pd.DataFrame(M.values)
    else:
        Mat = pd.DataFrame(weights.values)

    M_quant = np.abs(Mat.replace(0,np.nan))
    quant = np.nanquantile(M_quant, q=q)
    qMasked = M[M_quant >= quant].replace(np.nan, 0)

    return qMasked
