# ps2_functions.py
# Jay S. Stanley III, Yale University, Fall 2018
# CPSC 453 -- Problem Set 2
#
# This script contains functions for implementing graph clustering and signal processing.
#

import numpy as np
import codecs
import json
import math
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import eigh
from sklearn.cluster import KMeans

def load_json_files(file_path):
    '''
    Loads data from a json file

    Inputs:
        file_path   the path of the .json file that you want to read in

    Outputs:
        my_array    this is a numpy array if data is numeric, it's a list if it's a string

    '''

    #  load data from json file
    with codecs.open(file_path, 'r', encoding='utf-8') as handle:
        json_data = json.loads(handle.read())

    # if a string, then returns list of strings
    if not isinstance(json_data[0], str):
        # otherwise, it's assumed to be numeric and returns numpy array
        json_data = np.array(json_data)

    return json_data


def gaussian_kernel(X, kernel_type="gaussian", sigma=3.0, k=5):
    """gaussian_kernel: Build an adjacency matrix for data using a Gaussian kernel
    Args:
        X (N x d np.ndarray): Input data
        kernel_type: "gaussian" or "adaptive". Controls bandwidth
        sigma (float): Scalar kernel bandwidth
        k (integer): nearest neighbor kernel bandwidth
    Returns:
        W (N x N np.ndarray): Weight/adjacency matrix induced from X
    """
    _g = "gaussian"
    _a = "adaptive"

    kernel_type = kernel_type.lower()
    D = squareform(pdist(X))
    if kernel_type == "gaussian":  # gaussian bandwidth checking
        print("fixed bandwidth specified")

        if not all([type(sigma) is float, sigma > 0]):  # [float, positive]
            print("invalid gaussian bandwidth, using sigma = max(min(D)) as bandwidth")
            D_find = D + np.eye(np.size(D, 1)) * 1e15
            sigma = np.max(np.min(D_find, 1))
            del D_find
        sigma = np.ones(np.size(D, 1)) * sigma
    elif kernel_type == "adaptive":  # adaptive bandwidth
        print("adaptive bandwidth specified")

        # [integer, positive, less than the total samples]
        if not all([type(k) is int, k > 0, k < np.size(D, 1)]):
            print("invalid adaptive bandwidth, using k=5 as bandwidth")
            k = 5

        knnDST = np.sort(D, axis=1)  # sorted neighbor distances
        sigma = knnDST[:, k]  # k-nn neighbor. 0 is self.
        del knnDST
    else:
        raise ValueError

    W = ((D**2) / sigma[:, np.newaxis]**2).T
    W = np.exp(-1 * (W))
    W = (W + W.T) / 2  # symmetrize
    W = W - np.eye(W.shape[0])  # remove the diagonal
    return W


# BEGIN PS2 FUNCTIONS


def sbm(N, k, pij, pii, sigma):
    """sbm: Construct a stochastic block model

    Args:
        N (integer): Graph size
        k (integer): Number of clusters
        pij (float): Probability of intercluster edges
        pii (float): probability of intracluster edges

    Returns:
        A (numpy.array): Adjacency Matrix
        gt (numpy.array): Ground truth cluster labels
        coords(numpy.array): plotting coordinates for the sbm
    """
    
    p_matrix = np.random.uniform(0,1,(k,k))
    gt = np.append(np.random.randint(1,k+1,(N // k) * k), 
                   np.array(range(1, (N % k) + 1)))

    A = np.zeros((N,N))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            cluster_1 = gt[i]
            cluster_2 = gt[j]

            if cluster_1 == cluster_2:
                p = np.random.uniform(0,1,1)
                if p <= pii:
                    A[i][j] = 1       
            else:
                p = np.random.uniform(0,1,1)
                if p <= pij:
                    A[i][j] = 1

    coords = np.zeros((N, 2))
    for idx, i in enumerate(gt):
        unit_circle_degree = 2 * np.pi / k * (i - 1)
        unit_circle_x = math.cos(unit_circle_degree)
        unit_circle_y = math.sin(unit_circle_degree)
        coords[idx, 0] = np.random.normal(unit_circle_x, sigma, 1)
        coords[idx, 1] = np.random.normal(unit_circle_y, sigma, 1)
    
    return A, gt, coords


def L(A, normalized=False):
    """L: compute a graph laplacian

    Args:
        A (N x N np.ndarray): Adjacency matrix of graph
        normalized (bool, optional): Normalized or combinatorial Laplacian

    Returns:
        L (N x N np.ndarray): graph Laplacian
    """
    
    n, m = A.shape
    D = np.diag(A.sum(axis=1).flatten())
    L = D - A
    if normalized == True:
        inverse_D = np.linalg.inv(D)**(1/2)
        L = inverse_D @ L @ inverse_D
    return L


def compute_fourier_basis(L):
    """compute_fourier_basis: Laplacian Diagonalization

    Args:
        L (N x N np.ndarray): graph Laplacian

    Returns:
        e (N x 1 np.ndarray): graph Laplacian eigenvalues
        psi (N x N np.ndarray): graph Laplacian eigenvectors
    """
    
    e, psi = eigh(L)
    return e, psi


def gft(s, psi):
    """gft: Graph Fourier Transform (GFT)

    Args:
        s (N x d np.ndarray): Matrix of graph signals.  Each column is a signal.
        psi (N x N np.ndarray): graph Laplacian eigenvectors
    Returns:
        s_hat (N x d np.ndarray): GFT of the data
    """
    s_hat = psi.T @ s
    return s_hat


def igft(s_hat, psi):
    s = psi @ s_hat
    return s


def filterbank_matrix(psi, e, h):
    """filterbank_matrix: build a filter matrix using the input filter h

    Args:
        psi (N x N np.ndarray): graph Laplacian eigenvectors
        e (N x 1 np.ndarray): graph Laplacian eigenvalues
        h (function handle): A function that takes in eigenvalues
        and returns values in the interval (0,1)

    Returns:
        H (N x N np.ndarray): Filter matrix that can be used in the form
        filtered_s = H@s
    """
    H = psi @ np.diag(h(e)) @ psi.T
    return H


def kmeans(X, k, nrep=5, itermax=300):
    """kmeans: cluster data into k partitions

    Args:
        X (n x d np.ndarray): input data, rows = points, cols = dimensions
        k (int): Number of clusters to partition
        nrep (int): Number of repetitions to average for final clustering 
        itermax (int): Number of iterations to perform before terminating
    Returns:
        labels (n x 1 np.ndarray): Cluster labels assigned by kmeans
    """
    dist_mat = np.zeros((k, X.shape[0]))
    n_points = X.shape[0]  # get the number of points in the data
    within_cluster_dist = np.zeros(nrep)
    nrep_labels = np.zeros((n_points, nrep))
    
    for rep in range(nrep):
    
        init = kmeans_plusplus(X, k)  # find your initial centroids
        old_labels = np.random.randint(0, k, n_points) # randomly initialize old labels
        new_labels = old_labels.copy()
        num_same_assignment = 0 # number of same assignment

        # perform kmeans
        for iteration in range(itermax):
            if iteration % 100 == 0 and iteration > 0:
                print("iteration " + str(iteration))
                
            for c in range(0, k):
                
                #calculate the euclidean distance between each data point and each centroid 
                for n in range(n_points):
                    dist_mat[c, n] = np.linalg.norm(X[n, :]-init[c, :]) ** 2

            #label each point with distance to closest centroid
            new_labels = np.argmin(dist_mat, axis=0) # n dim
            if np.array_equal(new_labels, old_labels): #check if the assignment is the same as last cycle
                num_same_assignment = num_same_assignment + 1
                if num_same_assignment == 3: #finish clustering if same assignment 3 times in a roll
                    break
                else:
                    num_same_assignment = 0  #continue clustering if not same assignment
            
            #calculate new centroids
            for c in range(0, k):
                init[c, :] = np.mean(X[new_labels==c, :], axis=0)

            #use new_labels as old_labels for the next cycle
            nrep_labels[:, rep] = new_labels
            old_labels = new_labels
            
        within_cluster_dist[rep] = np.sum(np.mean(dist_mat, axis=1))
            
    # choose the repetition with smallest with cluster distance
    smallest_iteration = np.argmin(within_cluster_dist[rep])
    labels = nrep_labels[:, smallest_iteration]
    
    return labels


def kmeans_plusplus(X, k):
    """kmeans_plusplus: initialization algorithm for kmeans
    Args:
        X (n x d np.ndarray): input data, rows = points, cols = dimensions
        k (int): Number of clusters to partition

    Returns:
        centroids (k x d np.ndarray): centroids for initializing k-means
    """
    
    centroids = [] 
    n_points = X.shape[0]
    centroids.append(X[np.random.randint(0, n_points, 1), :])
    
    ## compute remaining k - 1 centroids 
    for remaining_cluster in range(k - 1): 
        
        ## initialize stores distances of points to first centroid
        dist = np.zeros(n_points)
        for i in range(n_points):
            dist[i] = np.linalg.norm(X[i, :] - centroids[0])
              
        ## select data point with maximum distance as our next centroid 
        next_centroid = np.random.choice(np.arange(len(dist)), size = 1, p = dist / sum(dist))
        centroids.append(X[next_centroid, :])
    
    return np.array(centroids).squeeze()


def SC(L, k, psi=None, nrep=5, itermax=300, sklearn=False):
    """SC: Perform spectral clustering 
            via the Ng method
    Args:
        L (np.ndarray): Normalized graph Laplacian
        k (integer): number of clusters to compute
        nrep (int): Number of repetitions to average for final clustering
        itermax (int): Number of iterations to perform before terminating
        sklearn (boolean): Flag to use sklearn kmeans to test your algorithm
    Returns:
        labels (N x 1 np.array): Learned cluster labels
    """
    if psi is None:
        # compute the first k elements of the Fourier basis
        # use scipy.linalg.eigh
        
        e, psi = scipy.linalg.eigh(L)
        e_k = e[:k]
        psi_k = psi[:, :k]
        
    else:  # just grab the first k eigenvectors
        psi_k = psi[:, :k]

    # normalize your eigenvector rows
    psi_k_row_sums = np.linalg.norm(psi.copy(), axis=1)
    psi_norm = psi_k / psi_k_row_sums[:, np.newaxis]

    if sklearn:
        labels = KMeans(n_clusters=k, n_init=nrep, max_iter=itermax).fit_predict(psi_norm)
    else:
        labels = kmeans(psi_norm, k, nrep=nrep, itermax=itermax)
    return labels

