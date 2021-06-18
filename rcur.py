import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.cluster import spectral_clustering, KMeans, SpectralClustering
from itertools import permutations, combinations
import numpy.linalg as la
import scipy.io as sio
from collections import defaultdict, Counter
from sklearn.utils.extmath import randomized_svd
import time
import random
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import supervised
from scipy.sparse.linalg import svds


def get_mu_kappa(L, r):
    """
    Gets mu and kappa for incoherence
    Param: L, the input matrix
    Param: r, rank
    Return: [row incoherence, column incoherence], kappa, singular values
    """
    U, S, V = svds(L, r)
    V = V.T
    m, n = L.shape
    mu_U = max(np.sum(U**2, axis = 1)*m/r)
    mu_V = max(np.sum(V**2, axis = 1)*n/r)
    kappa = S[0]/S[r-1]
    return [mu_U, mu_V], kappa, S

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
        L = np.sum(sim, axis = 1)
        L[np.where(L == 0)] = np.inf
        L = np.diag(1/L)@sim
        u, _ = la.eig(L)
        u = np.sort(u)
        ncut /= (u[-n_clusters]-u[-n_clusters-1])
    return ncut

def compute_Leverage_Scores(matrix, rank):
    """
    Computes the leverage score
    Param: matrix, the similarity matrix
    Param: rank, the guessed rank
    Return: row leverage scores, column leverage scores
    """
    Uk, _, VTk = randomized_svd(matrix, n_components=rank)
    col_Lev_Scores = la.norm(VTk, axis = 0)**2/rank
    row_Lev_Scores = la.norm(Uk, axis = 1)**2/rank
    return row_Lev_Scores, col_Lev_Scores  

def compute_Row_Col_Scores(matrix):
    """
    Compute row scores for length sampling
    Param: matrix, the similarity matrix
    """
    fro = la.norm(matrix, 'fro')**2
    col_Scores = la.norm(matrix, axis = 0)**2/fro
    row_Scores = la.norm(matrix, axis = 1)**2/fro
    return row_Scores, col_Scores

def select_Rows_Cols(row_size, col_size, num_rows_select, num_cols_select, replace = False,
                     row_probabilities = None, col_probabilities = None, prob = None, rank = None):
    """
    Param: row_size, the number of rows
    Param: col_size, the number of columns
    Param: num_rows_select, number of rows to select
    Param: num_cols_select, number of columns to select
    Param: replace (default true) whether to select with replacement
    Param: row_probabilities (default None), the probability distribution for rows
    Param: col_probabilities (default None), the probability distribution for columns
    Param: prob (default None), the probability distribution for both rows and columns
    Param: rank (default None), the guessed rank for the similarity matrix
    Return: indices of selected rows, indices of selected columns
    """
    if row_probabilities is None and col_probabilities is None and prob is None:
        prob = 'uniform'    
    if row_probabilities is not None and col_probabilities is not None:
        prob = None
    if num_rows_select >= row_size:
        row_indices = np.arange(row_size)
    if num_cols_select >= col_size:
        col_indices = np.arange(col_size)
    if prob is None:
        row_indices = np.unique(np.random.choice(row_size, num_rows_select, replace = replace, p = row_probabilities))
        col_indices = np.unique(np.random.choice(col_size, num_cols_select, replace = replace, p = col_probabilities))
    elif prob == 'uniform':
        row_indices = np.unique(np.random.choice(row_size, num_rows_select, replace = replace))
        col_indices = np.unique(np.random.choice(col_size, num_cols_select, replace = replace))
    elif isinstance(prob,str):
        print("Invalid probability distribution")
    return row_indices, col_indices

def DEIM(singular_vector):
    """
    Param: truncated right singular vector matrix (size mxk)
    Return: k DEIM indices corresponding to column indices in {1,...,m}
    """
    v_temp = singular_vector[:,0]
    indices = [np.argmax(v_temp)]
    for i in range(1,len(singular_vector[0])):
        v_temp = singular_vector[:,i].reshape(len(singular_vector),1)
        if len(indices)>1:
            proj = la.inv(singular_vector[indices,:i])@v_temp[indices]
            residual = v_temp - singular_vector[:,:i]@proj
        else:
            proj = 1/singular_vector[indices,0]*v_temp[indices]
            residual = v_temp - singular_vector[indices,0]*proj
        indices.append(np.argmax(np.abs(residual)))
    return indices

def CUR(matrix, num_rows_select, num_cols_select, rank, row_prob = None, col_prob = None, replace = False, prob = 'uniform', row_indices = None, col_indices = None):
    #U not pseudoinv
    """
    Returns the CUR decomposition of a matrix. U is not pseudo-inverse
    Param: matrix, the similarity matrix
    Param: num_rows_select, the number of rows to select
    Param: num_cols_select, the number of columns to select
    Param: rank, the guessed rank of the similarity matrix
    Param: row_prob (default None), the probability distribution for rows
    Param: col_prob (default None), the probability distribution for columns
    Param: replace (default False), whether to select with replacement
    Param: prob (default 'uniform'), the probability distribution for both rows and columns
    Param: row_indices (defualt None), specifies the rows to use
    Param: col_indices (defualt None), specifies the columns to use
    Return: C, U, R, indices of row, indices of columns
    """
    if row_prob is not None and col_prob is not None:
        prob = None
    if prob == 'uniform':
        row_prob = None
        col_prob = None
    elif prob == 'leverage':
        row_prob, col_prob = compute_Leverage_Scores(matrix, rank)
    elif prob == 'length':
        row_prob, col_prob = compute_Row_Col_Scores(matrix)
    elif isinstance(prob,str):
        print("Invalid probability distribution")   
        
    if row_indices is None and col_indices is None:
        row_indices, col_indices = select_Rows_Cols(matrix.shape[0], matrix.shape[1], num_rows_select, 
                                                num_cols_select, replace = replace, row_probabilities = row_prob,
                                                col_probabilities = col_prob, prob = prob)
    elif row_indices is None:
        row_indices, _ = select_Rows_Cols(matrix.shape[0], matrix.shape[1], num_rows_select, 
                                                num_cols_select, replace = replace, row_probabilities = row_prob,
                                                col_probabilities = col_prob, prob = prob)
    elif col_indices is None:
        _, col_indices = select_Rows_Cols(matrix.shape[0], matrix.shape[1], num_rows_select, 
                                                num_cols_select, replace = replace, row_probabilities = row_prob,
                                                col_probabilities = col_prob, prob = prob)
    C = matrix[:,col_indices]
    R = matrix[row_indices,:]
    U = matrix[np.ix_(row_indices,col_indices)]
    return C, U, R, row_indices, col_indices

def RCUR(dataset, t, n_clusters, matrix_power = 4.5, ran = (5, 0), prob = 'uniform', col_mult = None):
    """
    Implements the RCUR algorithm and returns the best labels
    Param: dataset, the matrix whose columns are individual data points
    Param: t, how many initial similarity matrices to make to take the median of
    Param: n_clusters, the number of expected classes
    Param: matrix_power (default 4.5), the element-wise power for thresholding
    Param: ran (default (5, 0)), the range for guessed rank values
    Param: prob (defualt 'uniform'), the sampling algorithm to use 
    Param: col_mult (default None), the value for kappa. None implies kappa = infinity
    Return: the best clustering labels, the best similarity matrix, the lowest ncut, the best rank
    """
    minncut = np.inf
    bestcluster, bestsim, bestrank = 0, 0, 0
    row_prob, col_prob, row_indices, col_indices = None, None, None, None
    if prob=='length':
        row_prob, col_prob = compute_Row_Col_Scores(dataset)
    for rank in range(ran[0], ran[1]):
        if prob == 'leverage':
            row_prob, col_prob = compute_Leverage_Scores(dataset, rank)
        print(rank, end='')
        if col_mult is None:
            num_cols = len(dataset[0])
        else:
            num_cols = min(len(dataset[0]), col_mult*rank)
       
        if prob == 'deim':
            U, _, V = randomized_svd(dataset, n_components=rank)
            row_indices = DEIM(U)
            _, U, R, _, _ = CUR(dataset, rank, num_cols, rank, prob='uniform', row_indices = row_indices, col_indices = col_indices)  
            similarity_matrix = np.linalg.pinv(U)@R
            normY = la.norm(similarity_matrix,axis=0)
            normY[np.where(normY == 0)[0]] = 1
            similarity_matrix = similarity_matrix/normY
            similarity_matrix = np.abs(similarity_matrix.T @ similarity_matrix)**matrix_power
        else:
            X = np.zeros((t, len(dataset[0]), len(dataset[0])))
            for i in range(1, t):
                _, U, R, _, _ = CUR(dataset, rank, num_cols, rank, prob = prob, col_prob = col_prob, row_prob = row_prob)
                Y = np.linalg.pinv(U)@R
                normY = la.norm(Y,axis=0)
                normY[np.where(normY == 0)[0]] = 1
                Y = Y/normY
                X[i, :, :] = np.abs(Y.T @ Y)
            similarity_matrix = np.abs((np.median(X, axis=0)))**matrix_power
        cluster = spectral_clustering(similarity_matrix, n_clusters=n_clusters, assign_labels='discretize')
        nc = ncut(similarity_matrix, cluster, RSIM=False)
        if nc < minncut:
            minncut = nc
            bestrank = rank
            bestcluster = cluster
            bestsim = similarity_matrix
    return bestcluster, bestsim, minncut, bestrank
