# import required libraries
import numpy as np
import codecs, json


def load_json_files(file_path):
    '''
    Loads data from a json file

    Inputs:
        file_path   the path of the .json file that you want to read in

    Outputs:
        json_data    this is a numpy array if data is numeric, it's a list if it's a string

    '''

    #  load data from json file
    with codecs.open(file_path, 'r', encoding='utf-8') as handle:
        json_data = json.loads(handle.read())

    # if a string, then returns list of strings
    if not isinstance(json_data[0], str):
        # otherwise, it's assumed to be numeric and returns numpy array
        json_data = np.array(json_data)

    return json_data



def compute_distances(X):
    '''
    Constructs a distance matrix from data set, assumes Euclidean distance

    Inputs:
        X       a numpy array of size n x p holding the data set (n observations, p features)

    Outputs:
        D       a numpy array of size n x n containing the euclidean distances between points

    '''
    if (len(X.shape) != 2):
        print("please input a 2D matrix")
        return -1

    n, p = X.shape
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(p):
                s += (X[i,k] - X[j,k])**2 #Euclidean distance
            D[i, j] = s**0.5

    # return distance matrix
    return D


def compute_affinity_matrix(D, kernel_type, sigma=None, k=None):
    '''
    Construct an affinity matrix from a distance matrix via gaussian kernel.

    Inputs:
        D               a numpy array of size n x n containing the distances between points
        kernel_type     a string, either "gaussian" or "adaptive".
                            If kernel_type = "gaussian", then sigma must be a positive number
                            If kernel_type = "adaptive", then k must be a positive integer
        sigma           the non-adaptive gaussian kernel parameter
        k               the adaptive kernel parameter

    Outputs:
        W       a numpy array of size n x n that is the affinity matrix

    '''
    if (len(D.shape) != 2):
        print("D must be a 2D matrix")
        return -1

       
    W = np.zeros((D.shape[0], D.shape[1]))
    
    if (kernel_type == "gaussian"):
        if (sigma == None or sigma <= 0):
            print("sigma must be a positive number")
            return -1
        # Gaussian Kernel
        W = np.exp(-D**2 / sigma**2)

    elif (kernel_type == "adaptive"):
        if (k == None or k <= 0):
            print("k must be a positive integer")
            return -1
        
        # Calculate K nearest neighbor distance
        sigmak = np.sort(D, axis=1)[:, k]
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                # Adaptive Kernel distance based on sigmak
                W[i, j] = 0.5 * (np.exp(-D[i,j]**2 / sigmak[i]**2) + np.exp(-D[i,j]**2 / sigmak[j]**2))
    
    # return the affinity matrix
    return W


def diff_map_info(W):
    '''
    Construct the information necessary to easily construct diffusion map for any t

    Inputs:
        W           a numpy array of size n x n containing the affinities between points

    Outputs:

        diff_vec    a numpy array of size n x n-1 containing the n-1 nontrivial eigenvectors of Markov matrix as columns
        diff_eig    a numpy array of size n-1 containing the n-1 nontrivial eigenvalues of Markov matrix

        We assume the convention that the coordinates in the diffusion vectors are in descending order
        according to eigenvalues.
    '''
    
    if (len(W.shape) != 2):
        print("W must be a 2D matrix")
        return -1
    
    D=np.diag(np.sum(W, axis=0))
    M=np.linalg.inv(D)@W
    Ms=D**(1/2)@M@np.linalg.inv(D)**(1/2)
    
    # Eigendecomposition
    diff_eig, diff_vec = np.linalg.eigh(Ms)
    
    #Reverse Order of eigenvectors based on eigenvalues
    diff_eig = diff_eig[::-1]
    diff_vec = diff_vec[:, ::-1]
    
    #actual diffusion map eigenvectors
    diff_vec = np.linalg.inv(D) @ D**(1/2) @ diff_vec
    diff_vec = diff_vec / np.linalg.norm(diff_vec, axis=1)[:, np.newaxis]
    
    #remove steady state (largest) eigenvector/value pair
    diff_vec = diff_vec[:, 1:]
    diff_eig = diff_eig[1:]

    # return the info for diffusion maps
    return diff_vec, diff_eig


def get_diff_map(diff_vec, diff_eig, t):
    '''
    Construct a diffusion map at t from eigenvalues and eigenvectors of Markov matrix

    Inputs:
        diff_vec    a numpy array of size n x n-1 containing the n-1 nontrivial eigenvectors of Markov matrix as columns
        diff_eig    a numpy array of size n-1 containing the n-1 nontrivial eigenvalues of Markov matrix
        t           diffusion time parameter t

    Outputs:
        diff_map    a numpy array of size n x n-1, the diffusion map defined for t
    '''
    
    #construct diffusion maps
    diff_map = (np.diag(diff_eig**t) @ diff_vec.T).T

    return diff_map

