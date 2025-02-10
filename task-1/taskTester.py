# -----------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------
from task import kmeans_sklearn, kmeans, knn_classifier, approximate_nearest_neighbour
import numpy as np
import time


N = 1000 # number of data points
D = 100 # dimension of data points
A = np.random.randn(N, D) # dataset
X = np.random.randn(D) # query point
K = 10 # number of nearest neighbours

# Example
def test_kmeans():
    kmeans_result = kmeans(N, D, A, K)
    print(f"KMeans result: {kmeans_result}")
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

    centroids = kmeans_sklearn(N, D, A, K)
    print(f"Centroids: {centroids}")
    #knn_result = test_knn()
    #ann_result = test_ann()

    #recall_rate = recall_rate(knn_result, ann_result)
    #print(f"Recall rate: {recall_rate * 100}%")
