import json
import random
import numpy as np


def cluster_points(X, mu):
    """
    Distribute data points into clusters.

    Inputs:
    - X: A list of data points in 2d space, each elements is a list of 2
    - mu: A list of K cluster centers, each elements is a list of 2

    Returns:
    - clusters: A dict, keys are cluster index {1,2, ..., K} (int),
                value are the list of corresponding data points.
    """

    clusters = {}
    #print ("points::",X)
    #print ("prototypes:",mu)
    for data in X:
      data=np.array(data)
      minimum=100000
      nearest=1
      for i in range(len(mu)):
         mu[i]=np.array(mu[i])
         distance=np.linalg.norm(data-mu[i])
         #print ("dist::",distance)
         if distance <minimum:
            minimum=distance
            nearest=i+1
      #print ("nearest::",nearest)
      data=data.tolist()
      if nearest not in clusters.keys():
         clusters[nearest]=[]
         clusters[nearest].append(data)
      else:
         clusters[nearest].append(data)
         
         
        
    #print ("clusters::",clusters.keys())
      
         

    # you need to fill in your solution here

    return clusters


def reevaluate_centers(mu, clusters):
    """
    Update cluster centers.

    Inputs:
    - mu: A list of K cluster centers, each elements is a list of 2
    - clusters: A dict, keys are cluster index {1,2, ..., K} (int),
                value are the list of corresponding data points.

    Returns:
    - newmu: A list of K updated cluster centers, each elements is a list of 2
    """
    newmu = []
    
    for k in clusters.keys():
      np_array=np.array(clusters[k])
      #print ("here::",np_array)
      mean=np.mean(np_array,axis=0)
      mean=mean.tolist()
      newmu.append(mean)
         

    # you need to fill in your solution here

    return newmu


def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])


def find_centers(X, K):
    # Initialize to K random centers
    random.seed(100)
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)

    return(mu, clusters)


def kmeans_clustering():
    # load data
    with open('hw4_circle.json', 'r') as f:
        data_circle = json.load(f)
    with open('hw4_blob.json', 'r') as f:
        data_blob = json.load(f)

    mu_all = {}
    clusters_all = {}
    for K in [2, 3, 5]:
        key = 'blob_K=' + str(K)
        mu_all[key], clusters_all[key] = find_centers(data_blob, K)
        key = 'circle_K=' + str(K)
        mu_all[key], clusters_all[key] = find_centers(data_circle, K)

    return mu_all, clusters_all


def main():
    mu_all, clusters_all = kmeans_clustering()

    print('K-means Cluster Centers:')
    for key, value in mu_all.items():
        print('\n%s:'% key)
        print(np.array_str(np.array(value), precision=4))

    with open('kmeans.json', 'w') as f_json:
        json.dump({'mu': mu_all, 'clusters': clusters_all}, f_json)


if __name__ == "__main__":
    main()
