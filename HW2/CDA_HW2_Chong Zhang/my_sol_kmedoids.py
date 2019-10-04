#% Your goal of this assignment is implementing your own K-medoids.
#% Please refer to the instructions carefully, and we encourage you to
#% consult with other resources about this algorithm on the web.
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

from scipy.spatial import distance
import numpy as np


def my_kmedoids(image_data, K):
    
    # randomly choose K data points as initial centroids
    centers = image_data[np.random.randint(low = 0, high =image_data.shape[0], size=K),:]
    

    # create a empty array to store cluster info
    image_cluster = np.zeros(image_data.shape[0]).astype(int)

    def update_centers(image_data, centers, image_cluster): #update the coordinates of the new centroids
        for i in range(K):
            new_center = np.mean(image_data[image_cluster==i],axis=0)# get the mean coordinates for each cluster (cluster center)
            centers[i] = new_center #store those mean coordinates for each cluster (cluster center)
            
        dist = distance.cdist(image_data, centers, 'cityblock') # calculate the Manhattan distance between each data point and each clsuter center
        for i in range(K):
            index = dist[:,i].argmin() # find the index for the closest data point to the cluster center
            centers[i] = image_data[index] # use the coordinates of the closest point as the medoids
        return centers
    
    def dist_cal(image_cluster, centers): # calculate the distance between each data point to each medoid
                                          # then find the closest medoid, and store that info.
        
        dist = distance.cdist(image_data,centers,'cityblock') #calculate the Manhattan distance
        for i in range(len(image_cluster)): #find the closest medoid, and store that info
            image_cluster[i] = dist[i].argmin()
        
        centers = update_centers(image_data,centers, image_cluster)
        return image_cluster, centers
    
    n =0
    
    while n<=300: # 300 iteration max
        centers_old = centers.copy() # hold the cluster info before update
        
        image_cluster, centers_new = dist_cal(image_cluster, centers)

        v = centers_new == centers_old # compare the updated and unupdated centroid coordinates
        if v.all() == False: # if not the same, then continue
            centers = centers_new
        else: # if the same, stop
             centers = centers_new
             break
        n += 1
    print ("Total iteration for k-mediods:", n)
    return image_cluster, centers
