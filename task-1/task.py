import numpy as np

# ------------------------------------------------------------------------------------------------
# Importing JAX for GPU acceleration
# ------------------------------------------------------------------------------------------------
from jax import numpy as jnp
from jax import jit

# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------


# Distance Functions

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

@jit
def distance_manhattan_jax(X, Y):
    return jnp.abs(X - Y)

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# Standard KMeans Algorithm

def calculate_centroids(clusters):
    # Calculate mean of all points in cluster
    return [np.mean(cluster, axis=0) for cluster in clusters]


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def kmeans_sklearn(N, D, A, K):
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(A)
    return kmeans.cluster_centers_

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

    return centroids

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# KNN Classifier

def knn_classifier(num_vectors, vector_dimension , dataset, query, k, distance_function=distance_l2):

    # Calculate distance between query and all vectors in dataset
    distances = [distance_function(query, vector) for vector in dataset]

    # Get top k nearest vectors
    nearest_vectors = np.argsort(distances)[:k]

    return nearest_vectors


# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# Approximate Nearest Neighbour Classifier

def approximate_nearest_neighbour(N, D, A, X, K):
    # Build up voronoi diagram using kmeans algorithm and then only calculate distance for the points in the same voronoi cell

    num_clusters = int(np.sqrt(N))  # Rule of thumb: sqrt(N) clusters
    # Calculate centroids using kmeans
    centroids = kmeans(N, D, A, num_clusters)

    # Calculate distance between query and all centroids
    distances_to_centroids = [distance_l2(X, centroid) for centroid in centroids]

    # Get the nearest 3 centroids
    nearest_centroids = np.argsort(distances_to_centroids)[:3]

    # Only find the nearest neighbours for the points in the two nearest centroids
    candidate_points = []
    for centroid in nearest_centroids:
        candidate_points.extend(knn_classifier(N, D, A, centroids[centroid], K))

    # select top K nearest vectors
    distances = [distance_l2(X, vector) for vector in A[candidate_points]]
    nearest_vectors = np.argsort(distances)[:K]

    return nearest_vectors