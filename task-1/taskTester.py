# -----------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------
from task import kmeans, knn_classifier, approximate_nearest_neighbour
import numpy as np

# use random data
N = 1000
D = 100
A = np.random.randn(N, D)
X = np.random.randn(D)
K = 10

# Example
def test_kmeans():
    kmeans_result = kmeans(N, D, A, K)
    print(kmeans_result)


def test_knn():
    knn_result = knn_classifier(N, D, A, X, K)
    print(knn_result)


def test_ann():
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
