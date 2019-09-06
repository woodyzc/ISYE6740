import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt 

food  = pd.read_csv("food-consumption.csv")

# drop Sweden, Finland and Spain
remove = ['Sweden', "Finland", "Spain"]
food = food.loc[-food["Country"].isin(remove),:]

# store all countries' name
Country = food["Country"]

# minors the mean for each column.
food = food[food.columns.difference(['Country'])] - np.mean(food[food.columns.difference(['Country'])], axis=0)

# get the covariance matrix
food_cov = np.cov(food,rowvar=False)

# calculate the eigenvectors and eigenvalues of the covariance matrix
vals, vecs = np.linalg.eig(food_cov)

# sort them from high to low
vals = vals[np.argsort(-vals)]
vecs = vecs[:,np.argsort(-vals)]

# get principle component 1 and 2
PC1 = food.dot(vecs[:,0].transpose()).real
PC2 = food.dot(vecs[:,1].transpose()).real


PCA = pd.DataFrame(data={"Country":Country, "PC1":PC1, "PC2":PC2}).reset_index()

# plot principle component 1 and 2, then label each data points
p1 = sns.regplot(x='PC1',y='PC2', data=PCA,fit_reg=False)
for line in range(0,PCA.shape[0]):
    p1.text(PCA.PC1[line]+1, PCA.PC2[line]+1, PCA.Country[line], horizontalalignment='left', size='medium', color='black')
plt.show()

