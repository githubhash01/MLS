import jax.debug
import numpy as np

# ------------------------------------------------------------------------------------------------
# Importing JAX for GPU acceleration
# ------------------------------------------------------------------------------------------------
from jax import numpy as jnp
from jax import jit
import jax.random as random
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


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def kmeans_sklearn(N, D, A, K):
    kmeans = KMeans(n_clusters=K, init='random')
    kmeans.fit(A)
    return kmeans.cluster_centers_

def kmeans(N, D, A, K, max_iter=200):
    """
    N: number of data points
    D: dimension of data points
    A: dataset
    K: number of clusters
    """
    # Set initial inertia to infinity
    inertia = np.inf
    # Randomly generate K centroids
    centroids = A[np.random.choice(N, K, replace=False)]

    for _ in range(max_iter):

        # Assign points to clusters
        clusters = [
            [x for x in A if np.argmin([distance_l2(x, centroid) for centroid in centroids]) == j]
            for j in range(K)
        ]
        # Update centroids
        new_centroids = [
            np.mean(cluster, axis=0) if len(cluster) > 0 else A[np.random.randint(N)]
            for cluster in clusters
        ]

        # Calculate new inertia
        new_inertia = sum([
            np.sum([distance_l2(x, new_centroids[j]) for x in cluster])
            for j, cluster in enumerate(clusters)
        ])

        print(f"Iteration {_}: Inertia = {new_inertia}")

        # Check for inertial convergence
        if np.isclose(inertia, new_inertia, atol=1e-5):
            print("Converged")
            return new_centroids

        inertia = new_inertia
        centroids = new_centroids

    print("Did not converge")
    return centroids

def kmeans_jax(N, D, A, K, max_iter=200):

    key = random.PRNGKey(0)
    # Create an immutable tensor C of shape K x D with random values
    #centroids = random.uniform(key, (K, D))  # C: K x D
    # create an immutable tensor C of shape K x D with random values using choice
    centroids = A[random.choice(key, N, (K, D), replace=False)]  # C: K x D
    # Initialize assignments as zeros
    assignments = jnp.zeros(N, dtype=jnp.int32)  # Shape: (N,)

    def exit_condition(state):
        i, centroids, assignments = state
        return i < max_iter  # Stop when max_iter is reached

    def update_step(state):
        i, centroids, assignments = state

        # Compute distances & assignments
        distances = jnp.sum((A[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)  # N x K
        new_assignments = jnp.argmin(distances, axis=1)  # N

        def compute_inertia(A, assignments, centroids):
            # Get the centroid corresponding to each point's assignment
            assigned_centroids = centroids[assignments]  # Shape: (N, D)

            # Compute squared L2 distance for all points
            distances = jnp.sum((A - assigned_centroids) ** 2, axis=1)  # Shape: (N,)

            # Sum over all points to get total inertia
            return jnp.sum(distances)

        def compute_new_centroids(A, assignments, K):
            # Sum of points in each cluster
            cluster_sums = jnp.zeros((K, A.shape[1]))
            cluster_sums = cluster_sums.at[assignments].add(A)

            # Count of points in each cluster
            cluster_counts = jnp.bincount(assignments, length=K).astype(jnp.float32).reshape(-1, 1)

            # Avoid division by zero (if a cluster has no points)
            cluster_counts = jnp.where(cluster_counts == 0, 1, cluster_counts)

            # Compute new centroids
            return cluster_sums / cluster_counts  # Shape: (K, D)

        # Compute new centroids
        new_centroids = compute_new_centroids(A, new_assignments, K)  # K x D

        inertia = compute_inertia(A, new_assignments, new_centroids)

        jax.debug.print("Iteration {i}: Inertia = {inertia}", i=i, inertia=inertia)

        return i + 1, new_centroids, new_assignments  # Update state

    i_init = 0
    final_state = lax.while_loop(exit_condition, update_step, (i_init, centroids, assignments))

    final_iter, final_centroids, final_assignments = final_state
    assert final_centroids.shape == (K, D), "Centroids shape is incorrect!"
    assert final_assignments.shape == (N,), "Assignments shape is incorrect!"

    return final_centroids







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