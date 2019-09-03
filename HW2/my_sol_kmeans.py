#% Your goal of this assignment is implementing your own K-means.
#%
#% Input:
#%     pixels: data set. Each row contains one data point. For image
#%     dataset, it contains 3 columns, each column corresponding to Red,
#%     Green, and Blue component.
#%
#%     K: the number of desired clusters. Too high value of K may result in
#%     empty cluster error. Then, you need to reduce it.
#%
#% Output:
#%     class: the class assignment of each data point in pixels. The
#%     assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
#%     of class should be either 1, 2, 3, 4, or 5. The output should be a
#%     column vector with size(pixels, 1) elements.
#%
#%     centroid: the location of K centroids in your result. With images,
#%     each centroid corresponds to the representative color of each
#%     cluster. The output should be a matrix with K rows and
#%     3 columns. The range of values should be [0, 255].
#%     
#%
#% You may run the following line, then you can see what should be done.
#% For submission, you need to code your own implementation without using
#% the kmeans matlab function directly. That is, you need to comment it out.

from sklearn.cluster import KMeans
import numpy as np

def my_kmeans(image_data, K):
    
    # randomly choose K data points as initial centroids
    centers = image_data[np.random.randint(low = 0, high =image_data.shape[0], size=K),:]
    

    # create a empty array to store cluster info
    image_cluster = np.zeros(image_data.shape[0]).astype(int)
    
    def update_centers(image_cluster): #update the coordinates of the new centroids
        for i in range(0,K):
            new_center = np.mean(image_data[image_cluster==i],axis=0)# get the mean coordinates of the centroids
            centers[i] = new_center
        return centers
    
    def dist_cal(image_cluster, centers): # calculate the distance between each data point to each center
                                          # then find the closest center, and store that info in the last column
        for i in range(image_cluster.shape[0]):
            dist = np.linalg.norm(image_data[i]-centers,axis=1) #calculate the distance between centroids and each point
            image_cluster[i] = dist.argmin() # choose the cloest one
        
        centers = update_centers(image_cluster)
        return image_cluster, centers
    
    n =0
    
    while n<=300: # 300 iteration max
        centers_old = centers.copy() # hold the cluster info before update
        
        image_cluster, centers_new = dist_cal(image_cluster, centers)

        v = centers_new == centers_old # compare the updated and unupdated centroid coordinates
        if v.all() == False: # if not the same, then continue
            centers = centers_new
        else: # if the same, stop
            n = 301
            centers = centers_new
        n += 1
    return image_cluster, centers
