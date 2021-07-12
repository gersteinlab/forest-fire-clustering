# forest-fire-clustering


Clustering Method Inspired by Forest Fire Dynamics

## Description

Here, we introduce Forest Fire Clustering, an efficient and interpretable clustering method for discovering and validating cell types in single-cell sequencing analysis. Different than the existing methods, our clustering algorithm makes minimal prior assumptions about the data and provides point-wise posterior probabilities for internal validation. Additionally, it computes point-wise label entropies that can highlight novel transition cell types de novo along developmental pseudo-time manifolds. Lastly, our inductive algorithm is able to make robust inferences in an online-learning context.

## Getting Started

### Dependencies

* python_requires='>=3.6',
    * numpy==1.18.1
    * scipy==1.4.1
    * scikit-learn==0.22.2.post1
    * numba==0.53.1


### Installing


```
pip install forest-fire-clustering
```


### Executing program

"X" is a sample x feature matrix

To generate clustering:

```
coord_ffc = FFC(X, sigma=0.1, k=300, num_permute=int(X.shape[0]/10))
coord_ffc.preprocess()
coord_ffc.fit(fire_temp=100)
```

To validate the results:

```
coord_ffc.validate()
oord_ffc.entropy()
coord_ffc.pval()

coord_ffc.entropy_list # list of entropies of the data point
coord_ffc.pval_list # list of posterior significance values

```


## Authors

Zhanlin Chen, Jeremy Goldwasser, Philip Tuckman, Jing Zhang, Mark Gerstein




