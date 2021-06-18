import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import spectral_clustering, KMeans, SpectralClustering
from itertools import permutations, combinations
import numpy.linalg as la
from collections import defaultdict, Counter
from sklearn.utils.extmath import randomized_svd
import time
import random
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import supervised
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix

def loss_val(A, bestcluster):
    """
    Calculates the percentage loss of clusters
    Param: A, the actual cluster labels
    Param: bestcluster, the predicted clustering labels
    Return: float, percentage clustering
    """
    labels_true, labels_pred = supervised.check_clusterings(np.array(A).flatten(), bestcluster)
    value = supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return (1- value[r, c].sum() / len(labels_true))*100

def ncut(sim, clusters, RSIM=True):
    """
    Calculates ncut
    Param: sim, the similarity matrix
    Param: clusters, the clustering labels for the columns
    Param: RSIM (default true), whether to divide ncut by the magnitude of a gap in eigenvalues according to "Shape Interaction Matrix Revisited and Robustified: Efficient Subspace Clustering with Corrupted and Incomplete Data"
    Return: ncut
    """
    clusterset = set(clusters)
    n_clusters = len(clusterset)
    ncut = 0.
    for c in clusterset:
        w = np.sum(sim[np.ix_(clusters==c,clusters!=c)])
        assoc = np.sum(sim[clusters==c,:])
        assoc2 = np.sum(sim[clusters!=c,:])
        ncut += w#/assoc + w/assoc2
    if RSIM:
        L =np.diag(1/np.sum(sim, axis = 1))@sim
        u, _ = la.eig(L)
        u = np.sort(u)
        ncut /= (u[-n_clusters]-u[-n_clusters-1])
    return ncut

def RSIM(dataset, k, n_clusters):
    """
    Implements the RSIM algorithm and returns the best labels
    Param: dataset, the matrix whose columns are individual data points
    Param: k, the guess for the range of rank. Example: (0, 5)
    Param: n_clusters, the number of expected classes
    Return: the best clustering labels, the best similarity matrix, the lowest ncut, the best rank
    """
    ncut_list = []
    minncut = np.inf
    bestcluster, bestsim, bestrank = 0, 0, 0
    for rank in range(k[0], k[1], 1):
        print(rank, end=" ")
        svd = TruncatedSVD(rank)
        svd.fit_transform(dataset)
        v = svd.components_
        v = v / np.linalg.norm(v, axis=0)
        similarity_matrix = np.abs(v.T @ v)**4.5    # element-wise power 4.5 for thresholding
        cluster = spectral_clustering(similarity_matrix, n_clusters=n_clusters, assign_labels='discretize')
        nc = ncut(similarity_matrix, cluster, RSIM=True)
        ncut_list.append(nc)
        if nc < minncut:
            minncut = nc
            bestcluster = cluster
            bestsim = similarity_matrix
            bestrank = rank
    return bestcluster, bestsim, minncut, bestrank
