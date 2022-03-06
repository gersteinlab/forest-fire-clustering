import numpy as np
import scanpy as sc
import scipy
import sklearn
import anndata as ad
from multiprocessing import Pool

class FFC:
    
    """ 
    
    Forest Fire Clustering
    By Flynn Chen, Yale '20
    
    Example usage:
    ffc = FFC()
    ffc.preprocess(X)
    ffc.fit(fire_temp=100)
    ffc.cluster_labels
    
    """
    
    def __init__(self,
                 n_neighbors = 20,
                 fire_temp = 1, 
                 num_permute = 500,
                 method = "umap",
                 n_jobs = 1):
        
        self.n_neighbors = n_neighbors
        self.fire_temp = fire_temp
        self.num_permute = num_permute
        self.n_jobs = n_jobs
        self.method = method
        # "umap" or "gauss"
        # see scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html
        # for more options
        
    def preprocess(self, X, n_neighbors = 20):
        '''
        Inputs:
            X   [N x D np.ndarray]: Feature Matrix

        Outputs:
            A [N x n_neighbors np.ndarray]: Adjacency/Affinity Matrix
        '''
        
        self.X = ad.AnnData(X = X)
        
        # scanpy has very good neighborhood implementation
        # hard to beat and why rebuild the wheel?
        sc.pp.neighbors(self.X, 
                        n_neighbors=self.n_neighbors, 
                        method=self.method)
        self.A = self.X.obsp['connectivities']
        self.Dinv = np.array(1/self.A.sum(axis = 1)).flatten()
    
    def fit(self, A = None, fire_temp = None):
        '''
        Inputs:
            A   [N x n_neighbors np.ndarray]: Adjacency/Affinity Matrix

        Outputs:
            cluster_labels [N x 1 np.ndarray]: Output cluster labels
        '''

        if fire_temp is not None:
            self.fire_temp = fire_temp
            
        if A is not None:
            self.A = A
            
        # initialization
        n_points = self.A.shape[0] #the number of entries/data points
        cluster_labels = -np.ones(n_points) #a list of labels for each point

        #Dinv = self.Dinv #precompute all thresholds as inv degree
        self.A = self.A * self.fire_temp #precompute fire temperatures
        remaining_points = n_points #keep track of remaining points
        label_num = 0 #label number, j
        first_seed = True

        while (remaining_points > 0): #iterate until there is a label for each point
            if first_seed == True: # choose a random point to start a fire
                seed = np.random.randint(0, remaining_points) 
                first_seed = False
            else: # choose the point with the smallest heat as a heuristic
                seed = unlabel_pts_idx[np.argmin(heat)]

            cluster_labels[seed] = label_num
            unlabel_pts_idx = np.argwhere(cluster_labels == -1).flatten() #get int idx of unlabeled pts
            same_cluster_pts = (cluster_labels == label_num) #get bool idx of pts in the same cluster
            threshold = self.Dinv[unlabel_pts_idx] #get threshold for unlabeled pts

            burned = True
            while burned:
                heat = np.array(self.A[np.ix_(same_cluster_pts, \
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
        self.A = self.A * (1 / self.fire_temp) #uncompute fire temperatures
        return cluster_labels
    
    def validate_serial(self, num_permute = None, seed = 0):
        
        np.random.seed(seed)
        
        # initialization
        cluster_labels = -np.ones((self.A.shape[0], self.num_permute)) #a list of labels for each point
        Dinv = self.Dinv
        A = self.A * self.fire_temp #precompute fire temperatures
        
        for p in range(self.num_permute): #perform permutation test
            if p % 100 == 0:  
                print("MC iteration", p)
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
        
        same_cluster_pts = np.zeros((self.A.shape[0], seeds.shape[0]), dtype=np.int32)
        same_cluster_pts[seeds, np.arange(seeds.shape[0])] = 1
        
        for i in range(seeds.shape[0]):
            unlabel_pts_idx = np.argwhere(same_cluster_pts[:, i] == 0).flatten() #get int idx of unlabeled pts
            threshold = self.Dinv[unlabel_pts_idx] #get threshold for unlabeled pts

            burned = True
            while burned:
                
                # calculate the heat by fire_temp * affinity
                heat = np.array(self.A[np.ix_(same_cluster_pts[:, i] == 1,
                                              unlabel_pts_idx)].mean(axis=0)).flatten() 
                               
                burned_indx = heat > threshold # bool idx of burned pts
                burned = np.sum(burned_indx)
                same_cluster_pts[unlabel_pts_idx[burned_indx], i] = 1
                not_burned_idx = np.logical_not(burned_indx)
                unlabel_pts_idx = unlabel_pts_idx[not_burned_idx]
                threshold = threshold[not_burned_idx]
                
        return same_cluster_pts
    
    def validate_master(self, num_permute = None, n_jobs=None, seed = 0):
        
        np.random.seed(seed)
        
        # initialization
        
        
        seeds = np.random.choice(np.arange(self.A.shape[0]), size=self.num_permute)
        seeds_split = np.array_split(seeds, self.n_jobs)
        
        self.A *= self.fire_temp #precompute fire temperatures
        with Pool(self.n_jobs) as p:
            returned_list = p.map(self.validate_worker, seeds_split)
        self.A /= self.fire_temp

        self.MC_labels = np.concatenate(returned_list, axis=1)
        self.MC_labels[self.MC_labels == 0] = -1
        for idx, cluster in enumerate(self.cluster_labels[seeds]):
            burned = np.argwhere(self.MC_labels[:, idx] >= 0).flatten()
            self.MC_labels[burned, idx] = self.MC_labels[burned, idx] * cluster
    
    def validate(self, parallel=False, num_permute = None, n_jobs=None, seed = 0):
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
        
        if parallel == False:
            self.validate_serial(self.num_permute, seed = seed)
        else:
            self.validate_master(self.num_permute, n_jobs, seed = seed)
            
        self.entropy()
        self.pval()
        
    def predict_all(self, x):
        prev_size = len(self.cluster_labels)
        self.preprocess(np.concatenate([self.X.X, x])) #re-preprocess
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
                # calculate the heat by fire_temp * affinity
                heat = np.array(A[same_cluster_pts, new_point_idx].mean(axis=0)).flatten() 
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
            pval = 1 - np.nanmean(labeled_data == self.cluster_labels[i])
            self.pval_list[i] = pval
        self.pval_list = np.nan_to_num(self.pval_list)
        #return self.pval_list    