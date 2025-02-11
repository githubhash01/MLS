import jax.debug
import numpy as np

# ------------------------------------------------------------------------------------------------
# Importing JAX for GPU acceleration
# ------------------------------------------------------------------------------------------------
from jax import numpy as jnp
from jax import jit
import jax.random as random
from sklearn.cluster import KMeans
from jax import lax
import time
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


def kmeans_sklearn(N, D, A, K):
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(A)
    return kmeans.cluster_centers_

def kmeans(N, D, A, K, max_iter=1000):
    """
    N: number of data points
    D: dimension of data points
    A: dataset
    K: number of clusters
    """
    inertia = np.inf

    def kmeans_plus_plus_init(A, K):
        """K-means++ initialization for better centroid selection."""
        centroids = [A[np.random.choice(len(A))]]
        for _ in range(1, K):
            dists = np.min([np.linalg.norm(A - c, axis=1) for c in centroids], axis=0)
            probs = dists ** 2 / np.sum(dists ** 2)
            centroids.append(A[np.random.choice(len(A), p=probs)])
        return np.array(centroids)

    centroids = kmeans_plus_plus_init(A, K)

    for _ in range(max_iter):

        # Assign points to clusters
        labels = np.argmin(np.linalg.norm(A[:, np.newaxis] - centroids, axis=2), axis=1)
        clusters = [A[labels == j] for j in range(K)]

        # Update centroids using random point if cluster is empty
        new_centroids = [
            np.mean(cluster, axis=0) if len(cluster) > 0 else random.choice(A)
            for cluster in clusters
        ]

        new_inertia = sum([
            np.sum([distance_l2(x, new_centroids[j]) for x in cluster])
            for j, cluster in enumerate(clusters)
        ])

        print(centroids, new_inertia)

        """
        # Check for inertial convergence
        if np.abs(inertia - new_inertia) < 1e-2:
            print("Converged")
            break

        """
        if np.linalg.norm(np.array(new_centroids) - np.array(centroids), axis=1).max() < 1e-6:
            print("Converged")
            return new_centroids

        centroids = new_centroids
        inertia = new_inertia

    return centroids

def kmeans_jax(N, D, A, K, max_iter=200):

    key = random.PRNGKey(0)
    key, subkey = random.split(key)

    centroid_indices = random.choice(subkey, A.shape[0], shape=(K,), replace=False)
    centroids = A[centroid_indices]

    # Compute the distances matrix D of shape N x K (each row: distances from a point to all centroids)
    summed_distances = (A[:, None, :] - centroids[None, :, :]) ** 2 # Summed Distances: N x K x D
    distances = jnp.sum(summed_distances, axis=-1)  # Distances: N x K

    # Assign each point to the cluster of the nearest centroid and save in vector A: N where a_i is the index of the cluster of the i-th point
    assignments = jnp.argmin(distances, axis=1)  # A: N

    # Now update the centroids by computing the mean of all points in each cluster
    tally = jnp.bincount(assignments, length=K)
    tally = jnp.where(tally == 0, 1, tally)  # Prevent division by zero

    sums = jnp.zeros((K, D))
    sums = sums.at[assignments].add(A)

    centroids = sums / tally[:, None]

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