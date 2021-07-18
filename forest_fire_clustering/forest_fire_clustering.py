import numpy as np
import scipy
import sklearn
from numba import jit
from multiprocessing import Pool

class FFC:
    
    """ 
    
    Forest Fire Clustering (with sparse matrix acceleration)
    By Flynn Chen, Yale '20
    
    """
    
    def __init__(self, 
                 X, 
                 fire_temp=1, 
                 sigma=0.15,
                 k = None,
                 num_permute = 200,
                 normalize_density=True,
                 n_jobs = 2):
        
        if (len(X.shape) != 2):
            print("X must be a 2D matrix")
            return -1
        
        self.fire_temp = fire_temp
        self.sigma = sigma
        self.num_permute = num_permute
        self.n_jobs = n_jobs
        self.X = X
        self.normalize_density = normalize_density
        
        if k is not None:
            self.k = k
        else:
            self.k = int(np.sqrt(X.shape[0]))
        
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True)
    def gaussian_kernel(D, sigma):

        # Gaussian Kernel
        A = np.exp(-D**2 / sigma**2)

        # return the affinity matrix
        return A
        
        
    def preprocess(self, sigma = None, n_jobs=None, k = None):
        
        if self.X.shape[0] < 2:
            print("cannot preprocess with less then 2 data points")
            return -1
        
        if sigma is not None:
            self.sigma = sigma
            
        if n_jobs is not None:
            self.n_jobs = n_jobs
            
        if k is not None:
            self.k = k
        X = sklearn.preprocessing.scale(self.X, axis=0)
        self.A = sklearn.neighbors.kneighbors_graph(X, \
                             int(self.k), \
                             mode='distance', \
                             include_self=True, \
                             n_jobs = self.n_jobs)
        self.A.data = self.gaussian_kernel(self.A.data, self.sigma)
        
        if self.normalize_density == True:
            D = scipy.sparse.diags(np.array(self.A.sum(axis = 0)).flatten(), 
                                   format = 'csc')
            D_inverse = scipy.sparse.linalg.inv(D)
            M = D_inverse @ self.A
            self.A = D.power(1/2) @ M @ scipy.sparse.linalg.inv(D).power(1/2)
        
        self.Dinv = np.array(1/self.A.sum(axis = 1)).flatten() #precompute all thresholds as inv degree

    def fit(self, fire_temp = None):
        '''
        Inputs:
            A   (N x N np.ndarray): Adjacency matrix of graph

        Outputs:
            cluster_labels (n x 1 np.ndarray): Output cluster labels
        '''

        if fire_temp is not None:
            self.fire_temp = fire_temp
        
        # initialization
        n_points = self.A.shape[0] #the number of entries/data points
        cluster_labels = -np.ones(n_points) #a list of labels for each point

        Dinv = self.Dinv
        A = self.A * self.fire_temp #precompute fire temperatures
        remaining_points = n_points #keep track of remaining points
        label_num = 0 #label number, j
        first_seed = True

        while (remaining_points > 0): #iterate until there is a label for each point
            print("points remaining after 1 cluster:", remaining_points)
            if first_seed == True: # choose a random point to start a fire
                seed = np.random.randint(0, remaining_points) 
                first_seed = False
            else: # choose the point with the smallest heat as a heuristic
                seed = unlabel_pts_idx[np.argmin(heat)]

            cluster_labels[seed] = label_num
            unlabel_pts_idx = np.argwhere(cluster_labels == -1).flatten() #get int idx of unlabeled pts
            same_cluster_pts = (cluster_labels == label_num) #get bool idx of pts in the same cluster
            threshold = Dinv[unlabel_pts_idx] #get threshold for unlabeled pts

            burned = True
            while burned:
                heat = np.array(A[np.ix_(same_cluster_pts, \
                                         unlabel_pts_idx)] \
                                .mean(axis=0)).flatten() # calculate the heat by fire_temp * affinity
                burned_indx = heat > threshold # bool idx of burned pts
                burned = np.sum(burned_indx)
                same_cluster_pts[unlabel_pts_idx[burned_indx]] = 1
                not_burned_idx = np.logical_not(burned_indx)
                unlabel_pts_idx = unlabel_pts_idx[not_burned_idx]
                threshold = threshold[not_burned_idx]

            cluster_labels[same_cluster_pts] = label_num
            remaining_points -= np.sum(same_cluster_pts)
            label_num = label_num + 1 # increment labels to burn the next cluster
        
        self.cluster_labels = cluster_labels
        return cluster_labels
    
    def validate_serial(self, num_permute = None):
        
        # input
        if num_permute is not None:
            self.num_permute = num_permute
        
        if self.num_permute > self.A.shape[0]:
            self.num_permute = self.A.shape[0]
        
        if len(self.cluster_labels) == 0:
            print("No fitting has been run yet.")
            return -1
        
        # initialization
        cluster_labels = -np.ones((self.A.shape[0], self.num_permute)) #a list of labels for each point
        Dinv = self.Dinv
        A = self.A * self.fire_temp #precompute fire temperatures
        
        for p in range(self.num_permute): #perform permutation test

            seed = np.random.randint(A.shape[0])
            label_num = self.cluster_labels[seed]
            cluster_labels[seed, p] = label_num
            unlabel_pts_idx = np.argwhere(cluster_labels[:, p] == -1).flatten() #get int idx of unlabeled pts
            same_cluster_pts = (cluster_labels[:, p] == label_num) #get bool idx of pts in the same cluster
            threshold = Dinv[unlabel_pts_idx] #get threshold for unlabeled pts

            burned = True
            while burned:
                heat = np.array(A[np.ix_(same_cluster_pts, \
                                         unlabel_pts_idx)] \
                                .mean(axis=0)).flatten() # calculate the heat by fire_temp * affinity
                burned_indx = heat > threshold # bool idx of burned pts
                burned = np.sum(burned_indx)
                same_cluster_pts[unlabel_pts_idx[burned_indx]] = 1
                not_burned_idx = np.logical_not(burned_indx)
                unlabel_pts_idx = unlabel_pts_idx[not_burned_idx]
                threshold = threshold[not_burned_idx]

            cluster_labels[same_cluster_pts, p] = label_num
            
        self.MC_labels = cluster_labels
        return cluster_labels
    
    def validate_worker(self, seeds):
        
        A = scipy.sparse.load_npz("A.npz")
        Dinv = np.load("Dinv.npy")
        
        same_cluster_pts = np.zeros((A.shape[0], seeds.shape[0]), dtype=np.int32)
        
        for i in range(seeds.shape[0]):
            same_cluster_pts[seed, i] = 1
            unlabel_pts_idx = np.argwhere(same_cluster_pts[:, i] == 0).flatten() #get int idx of unlabeled pts
            threshold = Dinv[unlabel_pts_idx] #get threshold for unlabeled pts

            burned = True
            while burned:
                heat = np.array(A[np.ix_(same_cluster_pts[:, i], unlabel_pts_idx)].mean(axis=0)).flatten() # calculate the heat by fire_temp * affinity
                burned_indx = heat > threshold # bool idx of burned pts
                burned = np.sum(burned_indx)
                same_cluster_pts[unlabel_pts_idx[burned_indx], i] = 1
                not_burned_idx = np.logical_not(burned_indx)
                unlabel_pts_idx = unlabel_pts_idx[not_burned_idx]
                threshold = threshold[not_burned_idx]

        return same_cluster_pts
    
    def validate_master(self, num_permute = None, n_jobs=None):
        
        # input
        if num_permute is not None:
            self.num_permute = num_permute
        
        if self.num_permute > self.A.shape[0]:
            self.num_permute = self.A.shape[0]
            
        if n_jobs is not None:
            self.n_jobs = n_jobs
        
        cluster_labels = self.cluster_labels
        if len(cluster_labels) == 0:
            print("No fitting has been run yet.")
            return -1
        
        # initialization
        Dinv = self.Dinv
        A = self.A * self.fire_temp #precompute fire temperatures
        
        scipy.sparse.save_npz("A.npz", A)
        np.save("Dinv.npy", Dinv)
        
        seeds = np.random.choice(np.arange(A.shape[0]), size=self.num_permute)
        seeds_jobs = np.split(seeds, self.n_jobs)
        
        with Pool(self.n_jobs) as p:
            print("sending job")
            returned_list = p.map(self.validate_worker, seeds_jobs)
    
        self.MC_labels = np.concatenate(returned_list)
        for idx, s in enumerate(seeds):
            self.MC_labels[:, idx] = self.MC_labels[:, idx] * s
        return cluster_labels
    
    def validate(self, parallel=False, num_permute = None, n_jobs=None):
        if parallel == False:
            self.validate_serial(num_permute)
        else:
            self.validate_master(num_permute, n_jobs)
        
    def predict_all(self, x):
        prev_size = len(self.cluster_labels)
        self.X = np.concatenate((self.X, x))
        self.preprocess() #re-preprocess
        self.cluster_labels = np.append(self.cluster_labels, -np.ones(x.shape[0]))
        A = self.A * self.fire_temp #precompute fire temperatures
        
        for i in range(len(x)):
            highest_heat = 0
            new_point_idx = prev_size + i
            threshold = self.Dinv[new_point_idx]
            existing_labels = np.unique(self.cluster_labels)[1:]
            self.cluster_labels[new_point_idx] = len(existing_labels)
            for u in existing_labels:
                same_cluster_pts = (self.cluster_labels == u)
                heat = np.array(A[same_cluster_pts, new_point_idx].mean(axis=0)).flatten() # calculate the heat by fire_temp * affinity
                #if heat > threshold:
                if heat > threshold and heat > highest_heat:
                    self.cluster_labels[new_point_idx] = u

        return self.cluster_labels[prev_size:]
    
    def entropy(self):
        if len(self.MC_labels) == 0:
            print("Error: Did not run Monte Carlo verification")
            return -1
        
        self.entropy_list = np.zeros(self.MC_labels.shape[0])
        for i in range(self.MC_labels.shape[0]): #iterate over every data point
            data_labels = self.MC_labels[i, :]
            labeled_data = data_labels[data_labels >= 0].astype(int)
            if len(labeled_data) == 0:
                self.entropy_list[i] = 0
            spread = np.bincount(labeled_data) / np.sum(np.bincount(labeled_data))
            node_entropy = scipy.stats.entropy(spread)
            self.entropy_list[i] = node_entropy
        self.entropy_list = np.nan_to_num(self.entropy_list)
        #return self.entropy_list
    
    def pval(self):
        if len(self.MC_labels) == 0:
            print("Error: Did not run Monte Carlo verification")
            return -1
        
        self.pval_list = np.zeros(self.MC_labels.shape[0])
        for i in range(self.MC_labels.shape[0]): #iterate over every data point
            data_labels = self.MC_labels[i, :]
            labeled_data = data_labels[data_labels >= 0].astype(int)
            if len(labeled_data) == 0:
                self.pval_list[i] = 0
            pval = 1 - np.mean(labeled_data == self.cluster_labels[i])
            self.pval_list[i] = pval
        self.pval_list = np.nan_to_num(self.pval_list)
        #return self.pval_list    
