# import torch
import cupy as cp
# import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def distance_cosine(X, Y):
    return 1 - (cp.dot(X, Y) / (cp.linalg.norm(X) * cp.linalg.norm(Y)))

def distance_l2(X, Y):
    return cp.sqrt(cp.sum((X - Y) ** 2))

def distance_dot(X, Y):
    return cp.dot(X, Y)

def distance_manhattan(X, Y):
    return cp.sum(cp.absolute(X - Y))

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn(N, D, A, X, K):
    """
    Input:
        N: Number of vectors
        D: Dimension of vectors
        A[N, D]: A collection of vectors
        X: A specified vector
        K: Top K
    """
    
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def our_kmeans(N, D, A, K):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K):
    pass

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example

def test_distances():
    cp.random.seed(12345)

    X = cp.random.rand(1000)
    Y = cp.random.rand(1000)

    cos_dist = distance_cosine(X, Y)
    print(f"Cosine Distance: {cos_dist}")

    l2_dist = distance_l2(X, Y)
    print(f"Euclidean Distance: {l2_dist}")

    dot_prod = distance_dot(X, Y)
    print(f"Dot Product: {dot_prod}")

    manhattan_dist = distance_manhattan(X, Y)
    print(f"Manhattan Distance: {manhattan_dist}")

    # Verify that arrays are using the GPU
    assert cp.get_array_module(cos_dist) is cp, "Cosine Distance is not computed on GPU"
    assert cp.get_array_module(l2_dist) is cp, "Euclidean Distance is not computed on GPU"
    assert cp.get_array_module(dot_prod) is cp, "Dot Product is not computed on GPU"
    assert cp.get_array_module(manhattan_dist) is cp, "Manhattan Distance is not computed on GPU"

    print("All tests passed! The code is running on GPU.")

def test_kmeans():
    N, D, A, K = testdata_kmeans("test_file.json")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("test_file.json")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)
    
def test_ann():
    N, D, A, X, K = testdata_ann("test_file.json")
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    test_distances()
