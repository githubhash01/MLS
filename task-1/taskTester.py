from task import kmeans, kmeans_jax, kmeans_jax, kmeans_sklearn
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import time

def generate_testing_data(N, D, K, cluster_std=0.5, random_state=42):
    """
    Generate synthetic dataset using sklearn's make_blobs and visualize the clusters.

    Parameters:
    - N: Number of samples
    - D: Number of dimensions
    - K: Number of clusters
    - cluster_std: Standard deviation of clusters (controls separation)
    - random_state: Seed for reproducibility

    Returns:
    - A: Generated dataset (NxD)
    - cluster_centers: The true cluster centers used for generation
    """
    A, labels, cluster_centers = make_blobs(
        n_samples=N,
        n_features=D,
        centers=K,
        cluster_std=cluster_std,
        random_state=random_state,
        return_centers=True
    )
    cluster_centers = np.array(cluster_centers).reshape(K, D)
    return A, cluster_centers



def test_accuracy(N, D, K, algorithm):
    """
    Generate test data, run both K-Means implementations, and plot the results.
    """
    A, true_centers = generate_testing_data(N, D, K)

    # baseline is using the sklearn implementation
    start = time.perf_counter()
    calculated_centroids_sklearn = kmeans_sklearn(N, D, A, K)
    end = time.perf_counter()
    print(f"Time taken by Sklearn: {end - start} seconds")

    # another test is using the numpy implementation

    start = time.perf_counter()
    calculated_centroids = kmeans(N, D, A, K)
    end = time.perf_counter()
    print(f"Time taken by Numpy: {end - start} seconds")

    # time the algorithm
    start = time.perf_counter()
    calculated_centroids = algorithm(N, D, A, K)
    end = time.perf_counter()
    print(f"Time taken by JAX: {end - start} seconds")

    plt.figure(figsize=(8, 6))
    plt.scatter(A[:, 0], A[:, 1], alpha=0.6, label="Data Points")
    plt.scatter(true_centers[:, 0], true_centers[:, 1], color='red', marker='X', s=200, label="True Centers")
    plt.scatter(calculated_centroids[:, 0], calculated_centroids[:, 1], color='green', marker='D', s=150, label="Custom KMeans")
    plt.legend()
    plt.title("Comparison of True Centers, Sklearn, and Custom K-Means")
    plt.show()

    return {
        "true_centers": true_centers,
        "calculated_centroids": calculated_centroids
    }

def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)


if __name__ == "__main__":

    # first generate the data
    # Parameters
    N = 10000  # number of data points
    D = 200 # dimension of data points
    K = 4 # number of clusters

    # Test KMeans
    print("Testing KMeans")
    test_accuracy(N, D, K, kmeans_jax)