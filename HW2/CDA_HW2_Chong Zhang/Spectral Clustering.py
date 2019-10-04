
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

data = pd.read_csv("nodes.txt", sep="\t", header = None)
edge = pd.read_csv("edges.txt", sep="\t", header = None)

nan_data = data[2][np.isnan(data[2])]

print ("here are some data points labelled NaN\n", nan_data)


# get the list of both origin and destination
node_from = edge[0]
node_to = edge[1]

# initiate a matrix with dismension 1490x1490
A = np.zeros((1490,1490))

# update the adjacency matrix based on edge info
for i in range(edge.shape[0]): 
    A[node_from[i]-1,node_to[i]-1] += 1

# calculate the diagonal matrix
D = np.diag(A.sum(axis=1))

# get the Laplacian matrix
L = D - A

# calculate the eigenvalues and eigenvectors of L 
vals, vecs = np.linalg.eig(L)

# sort the eigenvalues and eigenvectors from small to large
vals = vals[np.argsort(vals)]
vecs = vecs[:,np.argsort(vals)]


# perform Kmeans with k=2, using the first 2 eigenvectors
kmeans = KMeans(n_clusters=2,random_state=6).fit(vecs[:,0:2].real)
label = kmeans.labels_

# remove NaN data
true_label = data[2][-np.isnan(data[2])]
label = np.delete(label,[55, 110])

false_classification_rate =  1 - sum(label == true_label)/len(true_label)
print ("The  false classification rate is:", false_classification_rate)
