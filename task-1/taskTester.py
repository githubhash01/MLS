# -----------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------
from task import kmeans_sklearn, kmeans, knn_classifier, approximate_nearest_neighbour, kmeans_jax
import numpy as np
import time


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt



def generate_testing_data(N, D, K):
    """
    Generate synthetic dataset with well-separated clusters
    """
    #A, true_labels = make_blobs(n_samples=N, n_features=D, centers=K, cluster_std=0.5, random_state=42)
    cluster_centers = np.array([[5, 5], [-5, -5], [5, -5]])
    A = np.vstack([center + 0.5 * np.random.randn(N // K, D) for center in cluster_centers])
    # Visualize the dataset
    #plt.scatter(A[:, 0], A[:, 1])
    #plt.title("Manually Defined Clusters")
    #plt.show()
    return A, cluster_centers

# Example
def test_kmeans():
    kmeans_result = kmeans(N, D, A, K)
    return kmeans_result

def test_knn():
    # time the function
    start_time = time.perf_counter()
    knn_result = knn_classifier(N, D, A, X, K)
    end_time = time.perf_counter()
    print(f"Time taken KNN: {end_time - start_time}")
    return knn_result

def test_ann():
    start_time = time.perf_counter()
    ann_result = approximate_nearest_neighbour(N, D, A, X, K)
    end_time = time.perf_counter()
    print(f"Time taken ANN: {end_time - start_time}")
    return ann_result

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
    N = 100  # number of data points
    D = 2  # dimension of data points
    K = 3  # number of clusters

    A, true_centroids = generate_testing_data(N, D, K)
    print("True centroids")
    print(true_centroids)

    centroids_sklearn = kmeans_sklearn(N, D, A, K)
    print("Sklearn done")
    print(centroids_sklearn)

    centroids = kmeans(N, D, A, K)
    print("Custom done")
    print(centroids)

    """
    centroids = kmeans_sklearn(N, D, A, K)
    # order the centroids in the same order
    ordered_centroids = sorted(centroids, key=lambda x: np.sum(x))
    custom_centroids = test_kmeans()
    ordered_custom_centroids = sorted(custom_centroids, key=lambda x: np.sum(x))

    print(len(centroids), len(custom_centroids))

    # Compare the centroids by printing the distance for each of the centroids
    for centroid in zip(ordered_centroids, ordered_custom_centroids):
        print(np.linalg.norm(centroid[0] - centroid[1]))
    



    #centroids_jax = kmeans_jax(N, D, A, K)
    #print("Jax done")
    #print(centroids_jax)
    """