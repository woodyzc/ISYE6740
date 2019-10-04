import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px


food  = pd.read_csv("food-consumption.csv")

# drop Sweden, Finland and Spain
remove = ['Sweden', "Finland", "Spain"]
food = food.loc[-food["Country"].isin(remove),:]
food_list = food.columns[1::]

# store all countries' name
Country = food["Country"]

# minors the mean for each column.
food = food.iloc[:,1:]- np.mean(food.iloc[:,1:], axis=0)

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

# combine PC1 and PC1 with Country names
PCA = pd.DataFrame(data={"Country":Country, "PC1":PC1, "PC2":PC2}).reset_index()
PCA = PCA.iloc[:,1:4]

# perform Kmeans on PC1 and PC2 
K_means = KMeans(n_clusters=3).fit(PCA.iloc[:,1:3])
PCA["prediction"] = K_means.labels_.astype(object)


n=20 # number of food

PCs = vecs[:,0:2].real# obtain the first two eigenvectors (the directions)

# Draw a scatter plot of two-dimensional reduced representation for each country
p1 = sns.regplot(x='PC1', y='PC2', data=PCA, fit_reg=False)
for line in range(0,PCA.shape[0]):
    p1.text(PCA.PC1[line]+2, PCA.PC2[line]+2, PCA.Country[line], horizontalalignment='left', size='medium', color='black')
    
# add the projections of the original variables (food) on to the principal components, the red lines.
for i in range(n):
    plt.arrow(0, 0, PCs[i,0]*200, PCs[i,1]*200,color = 'r',alpha = 0.5)
    p1.text(PCs[i,0]*200, PCs[i,1]*200, food_list[i], color = 'g', ha = 'center', va = 'center')
        

plt.show()


# visulize K-means data
p2  = px.scatter(PCA, x='PC1',y='PC2',text="Country", color="prediction")
p2.update_traces(textposition='top center')
p2.show()

