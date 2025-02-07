import torch
#import cupy as cp
#import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
from collections import Counter

# ------------------------------------------------------------------------------------------------
# Importing JAX for GPU acceleration
# ------------------------------------------------------------------------------------------------
from jax import numpy as jnp
from jax import jit

# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

"""
Cosine Similarity: 

d(X, Y) = (X . Y) / (||X|| * ||Y||)

"""
def distance_cosine(X, Y):
    return np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))

@jit
def distance_cosine_jax(X, Y):
    return jnp.dot(X, Y) / (jnp.linalg.norm(X) * jnp.linalg.norm(Y))

"""
L2 Norm:

d(X, Y) = sqrt((X - Y)^2)

"""
def distance_l2(X, Y):
    return np.linalg.norm(X - Y)

@jit
def distance_l2_jax(X, Y):
    return jnp.linalg.norm(X - Y)

"""
Dot Product:

d(X, Y) = X . Y

"""
def distance_dot(X, Y):
    return np.dot(X, Y)

@jit
def distance_dot_jax(X, Y):
    return jnp.dot(X, Y)

"""
Manhattan Distance:

d(X, Y) = |X - Y|

"""

def distance_manhattan(X, Y):
    return np.abs(X - Y)

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn(N, D, A, X, K):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

# ------------------------------------------------------------------------------------------------
# Standard KMeans Algorithm
# ------------------------------------------------------------------------------------------------
"""
KMeans Algorithm

Input:

N: Number of vectors
D: Dimension of vectors
A[N, D]: A collection of vectors
X: A specified vector
K: Top K

Returns:
centroids - List of centroids

Algorithm Description:

1. Initialize:
   - Randomly select K points from dataset as initial centroids

2. REPEAT:
   a. Assignment step:
      - For each data point:
         - Calculate distance to each centroid
         - Assign point to the closest centroid's cluster

   b. Update step:
      - For each cluster:
         - Calculate mean of all points in cluster
         - Set new centroid position to cluster mean

3. UNTIL:
   - Centroids no longer move significantly OR
   - Maximum iterations reached

"""
def calculate_centroids(clusters):
    # Calculate mean of all points in cluster
    return [np.mean(cluster, axis=0) for cluster in clusters]

def kmeans(num_vectors, vector_dimension, dataset, num_clusters, distance_function=distance_l2, threshold=1e-5, max_iterations=1000):
    """
    Input:

    dataset_size: number of vectors
    vector_dimension: dimension of vectors
    dataset[dataset_size, vector_dimension]: A collection of vectors
    num_clusters: number of clusters
    Output:

    Result[dataset_size]: cluster ID for each vector
    """

    # randomly select num_clusters points from dataset as initial centroids
    centroids = dataset[np.random.choice(num_vectors, num_clusters, replace=False)] # central points for each cluster
    clusters = [[] for _ in range(num_clusters)] # list of lists of points in each cluster

    converged = False

    while not converged:

        # Assignment step
        for point in dataset:
            distances = [distance_function(point, centroid) for centroid in centroids]
            cluster_id = np.argmin(distances)
            clusters[cluster_id].append(point)

        # Update step - re-center the clusters
        new_centroids = calculate_centroids(clusters)
        # Check for convergence of values using threshold
        converged = np.allclose(centroids, new_centroids, rtol=threshold)

        # Check for maximum iterations
        if max_iterations == 0:
            break
        max_iterations -= 1

        centroids = new_centroids

        # Clear clusters
        clusters = [[] for _ in range(num_clusters)]

    return centroids

def knn_classifier(num_vectors, vector_dimension , dataset, query, k, distance_function=distance_l2):

    # Calculate distance between query and all vectors in dataset
    distances = [distance_function(query, vector) for vector in dataset]

    # Get top k nearest vectors
    nearest_vectors = np.argsort(distances)[:k]

    return nearest_vectors


# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def approximate_nearest_neighbour(N, D, A, X, K):
    # Build up voronoi diagram using kmeans algorithm and then only calculate distance for the points in the same voronoi cell

    # Calculate centroids using kmeans
    centroids = kmeans(N, D, A, K)

    # Calculate distance between query and all centroids
    distances = [distance_l2(X, centroid) for centroid in centroids]

    # Get the nearest 2 centroids
    nearest_centroids = np.argsort(distances)[:2]

    # Only find the nearest neighbours for the points in the two nearest centroids
    nearest_vectors = []
    for centroid in nearest_centroids:
        nearest_vectors.extend(knn_classifier(N, D, A, X, K, distance_function=distance_l2))

    # we only want the top K nearest vectors
    return nearest_vectors[:K]

# -----------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
def test_kmeans():
    N, D, A, K = testdata_kmeans("")
    #kmeans_result = our_kmeans(N, D, A, K)
    kmeans_result = kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("")
    knn_result = knn_classifier(N, D, A, X, K)
    print(knn_result)
    
def test_ann():
    N, D, A, X, K = testdata_ann("")
    ann_result = approximate_nearest_neighbour(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    test_kmeans()
    test_knn()
    test_ann()
