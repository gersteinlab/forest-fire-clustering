import numpy as np
import scipy
import sklearn
import random

import numpy as np
import scipy
import sklearn
import random

class FFC:
    
    def __init__(self, X, fire_temp=1, 
                 width=15, kernel_type="adaptive", 
                 distance_metric='euclidean', 
                 np_dist_ord=None, 
                 normalize_density=True,
                 normalize_affinity=True,
                 num_permute = 200):
        
        if (len(X.shape) != 2):
            print("X must be a 2D matrix")
            return -1
        
        self.fire_temp = fire_temp
        self.optimum_fire_temp = fire_temp
        self.width = width
        self.kernel_type = kernel_type
        self.distance_metric = distance_metric
        self.np_dist_ord = np_dist_ord
        self.output_labels = []
        self.entropy_list = []
        self.pval_list = []
        self.num_permute = num_permute
        self.MC_labels = []
        
        self.X = X
        self.D = sklearn.metrics.pairwise_distances(self.X, metric=self.distance_metric)
        
        if self.kernel_type == "adaptive":
            self.A = self.adaptive_gaussian_kernel(self.D, k=self.width)
        elif self.kernel_type == "gaussian":
            self.A = self.gaussian_kernel(self.D, sigma=self.width)
        else:
            self.A = self.D
        
        if normalize_density == True:
            D = np.diag(np.sum(self.A, axis=0))
            M = np.linalg.inv(D) @ self.A
            self.A = D**(1/2) @ M @ np.linalg.inv(D)**(1/2)
        
        if normalize_affinity == True:
            self.A = sklearn.preprocessing.normalize(self.A)
        

    def adaptive_gaussian_kernel(self, D, k):
        
        W = np.zeros((D.shape[0], D.shape[1]))
        
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
    
    def gaussian_kernel(self, D, sigma):
        
        if (sigma == None or sigma <= 0):
            print("sigma must be a positive number")
            return -1
        
        # Gaussian Kernel
        W = np.exp(-D**2 / sigma**2)

        # return the affinity matrix
        return W

    def fit(self, fire_temp = None):
        '''
        Forest Fire Clustering (by Flynn Chen & Jeremy Goldwasser)

        Inputs:
            A   (N x N np.ndarray): Adjacency matrix of graph

        Outputs:
            output_labels (n x 1 np.ndarray): Output cluster labels
        '''

        # input
        A = self.A
        if fire_temp == None:
            fire_temp = self.fire_temp
            
        self.optimum_fire_temp = fire_temp
        self.fire_temp = fire_temp
        
        # initialization
        n_points = A.shape[0] #the number of entries/data points
        n_label_list = -np.ones(n_points) #a list of labels for each point
        remaining_points = n_points
        num_label = 0
        first_flint = True
        centroids = []

        while (remaining_points > 0): #iterate until there is a label for each point
            print(remaining_points)
            if first_flint == True:
                flint = random.randint(0, remaining_points-1) #choose a random point to start a fire
                first_flint = False
            else:
                dist = []
                for i in range(n_points):
                    if n_label_list[i] >= 0:
                        dist.append(0)
                    else:
                        ## compute distance of 'point' from each of the previously 
                        ## selected centroid and store the minimum distance 
                        d = np.Inf
                        for j in range(len(centroids)): 
                            temp_dist = np.linalg.norm(A[i, :] - centroids[j], self.np_dist_ord) 
                            d = min(d, temp_dist) 
                        dist.append(d)

                ## select data point with probability 
                ## proportional to the current flint distance as our next centroid 
                dist = np.array(dist)
                flint = np.random.choice(np.arange(len(dist)), size = 1, p = dist / sum(dist))[0]
                dist = []

            remaining_points = remaining_points - 1
            n_label_list[flint] = num_label    
            centroids.append(A[flint, :])
            burning_list = [flint]

            while len(burning_list) != 0:
                flint_neighbors = A[burning_list[0], :] #point on fire
                for idx, neighbor_dist in enumerate(flint_neighbors):
                    if n_label_list[idx] == -1: #check if node is labeled
                        threshold = 1 / np.sum(A[idx, :]) #calculate the threshold by flash_point / degree
                        heat = np.mean(fire_temp * A[n_label_list == num_label, idx]) #calculate the heat by fire_temp * affinity
                        if heat > threshold:
                            burning_list.append(burning_list[0]) #reconsider neighbors of current burning node
                            burning_list.append(idx) #consider neighbors of current node's neighbor
#                             print(fire_temp, A[n_label_list == num_label, idx]) 
#                             print(" burned node ", idx)
#                             print(" " + str(burning_list[0]) + "-" + str(idx) + \
#                                 "|heat: " + str(heat) + \
#                                 "|threshold: " + str(threshold))
                            remaining_points = remaining_points - 1
                            n_label_list[idx] = num_label
                burning_list = burning_list[1:]

            num_label = num_label + 1

        # reorganize labels
        unique_labels = np.unique(n_label_list)
        output_labels = -np.ones(n_points)
        for idx, u in enumerate(unique_labels):
            output_labels[n_label_list==u] = idx

        self.output_labels = output_labels
        
        if len(unique_labels) == 1 or len(unique_labels) >= len(output_labels):
            return 1
        
        #### silhouette score ###
        else:
            return -1 * sklearn.metrics.silhouette_score(self.D, output_labels, metric = "precomputed")
        
        ### davies_bouldin_score ###
#         else:
#             return sklearn.metrics.davies_bouldin_score(self.X, output_labels)
    
        ### calinski_harabasz_score ###
#         else:
#             return -1 * sklearn.metrics.calinski_harabasz_score(self.X, output_labels)
        
    def annealing_fit(self):
        lw = [self.fire_temp / 100]
        up = [self.fire_temp * 100]
        ret = scipy.optimize.dual_annealing(self.fit, 
                                            bounds=list(zip(lw, up)),
                                            no_local_search = True)
        self.ret = ret
        self.optimum_fire_temp = self.ret.x[0]
        self.fit(self.optimum_fire_temp)
    
    def predict(self, x, add=True):
        x = np.expand_dims(x, axis=0)
        if add == True:
            self.X = np.concatenate((self.X, x))
            self.D = sklearn.metrics.pairwise_distances(self.X, metric = self.distance_metric)
            if self.kernel_type == "adaptive":
                self.A = self.adaptive_gaussian_kernel(self.D, k=self.width)
            elif self.kernel_type == "gaussian":
                self.A = self.gaussian_kernel(self.D, sigma=self.width)
            else:
                self.A = self.D

            unique_labels = np.unique(self.output_labels)
            threshold = 1 / np.sum(self.A[-1, :]) #calculate the threshold by flash_point / degree
            self.output_labels = np.append(self.output_labels, -1)
            np.random.shuffle(unique_labels)
            for label in unique_labels:
                heat = np.mean(self.fire_temp * self.A[self.output_labels == label, -1]) #calculate the heat by fire_temp * affinity
                if heat > threshold:
                    self.output_labels[-1] = label
            return self.output_labels[-1]
        
        else:
            X = np.concatenate((self.X, x))
            D = sklearn.metrics.pairwise_distances(X, metric=self.distance_metric)
            if self.kernel_type == "adaptive":
                A = self.adaptive_gaussian_kernel(D, k=self.width)
            elif self.kernel_type == "gaussian":
                A = self.gaussian_kernel(D, sigma=self.width)
            else:
                A = D

            unique_labels = np.unique(self.output_labels)
            threshold = 1 / np.sum(A[-1, :]) #calculate the threshold by flash_point / degree
            output_labels = np.append(self.output_labels, -1)
            np.random.shuffle(unique_labels)
            for label in unique_labels:
                heat = np.mean(self.fire_temp * A[output_labels == label, -1]) #calculate the heat by fire_temp * affinity
                if heat > threshold:
                    output_labels[-1] = label
            return output_labels[-1]
    
    def verify(self):
            
        # input
        output_labels = self.output_labels
        if len(output_labels) == 0:
            print("No fitting has been run yet.")
            return -1
        
        A = self.A
        fire_temp = self.optimum_fire_temp
        print("verifying with fire_temp:", fire_temp)
        num_permute = self.num_permute
        if num_permute > A.shape[0]:
            num_permute = A.shape[0]
        
        # initialization
        n_points = A.shape[0] #the number of entries/data points
        n_label_list = -np.ones((n_points, num_permute)) #a list of labels for each point

        for p in range(num_permute): #perform permutation test
            print(p)

            random_label_order = np.unique(output_labels)
            np.random.shuffle(random_label_order)
            for num_label in random_label_order:

                #randomly select a flint from the cluster
                cluster_node_idx = np.where(output_labels == num_label)[0]
                flint = np.random.choice(cluster_node_idx, 1)[0]

                #start propogation there
                n_label_list[flint, p] = num_label
                burning_list = [flint]

                while len(burning_list) != 0:

                    flint_neighbors = A[burning_list[0], :] #point on fire
                    for idx, neighbor_dist in enumerate(flint_neighbors):
                        if n_label_list[idx, p] == -1: #check if node is labeled
                            threshold = 1 / np.sum(A[idx, :]) #calculate the threshold by flash_point / degree
                            heat = np.mean(fire_temp * A[n_label_list[:, p] == num_label, idx]) #calculate the heat by fire_temp * affinity
                            if heat > threshold:
                                burning_list.append(burning_list[0]) #reconsider neighbors of current burning node
                                burning_list.append(idx) #consider neighbors of current node's neighbor
                                n_label_list[idx, p] = num_label

                    burning_list = burning_list[1:]

        self.MC_labels = n_label_list
        return n_label_list
    
    def entropy(self):
        if len(self.MC_labels) == 0:
            print("Error: Did not run Monte Carlo verification")
            return -1
        
        self.entropy_list = []
        for i in range(self.MC_labels.shape[0]): #iterate over every data point
            data_labels = self.MC_labels[i, :]
            labeled_data = data_labels[data_labels >= 0].astype(int)
            spread = np.bincount(labeled_data) / np.sum(np.bincount(labeled_data))
            node_entropy = scipy.stats.entropy(spread)
            self.entropy_list.append(node_entropy)
        return self.entropy_list
    
    def pval(self):
        if len(self.MC_labels) == 0:
            print("Error: Did not run Monte Carlo verification")
            return -1
        
        self.pval_list = []
        for i in range(self.MC_labels.shape[0]): #iterate over every data point
            data_labels = self.MC_labels[i, :]
            labeled_data = data_labels[data_labels >= 0].astype(int)
            pval = 1 - np.mean(labeled_data == self.output_labels[i])
            self.pval_list.append(pval)
        return self.pval_list
    
