import torch
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# Computes cosine distance between rows of X and Y using PyTorch.
# It normalizes each vector in X and Y using the L2 norm (torch.norm with p=2; here 'p' denotes the order of the norm, with p=2 corresponding to the Euclidean norm)
# and prevents division by zero with a small constant (1e-8).
# Then, it computes the cosine similarity via matrix multiplication (torch.matmul) and returns 1 - cosine similarity as a distance measure,
# where a value of 0 indicates identical directional vectors.
def distance_cosine(X, Y):
    X_norm = X / (torch.norm(X, p=2, dim=1, keepdim=True) + 1e-8)
    Y_norm = Y / (torch.norm(Y, p=2, dim=1, keepdim=True) + 1e-8)
    cos_sim = torch.matmul(X_norm, Y_norm.t())
    return 1 - cos_sim

# Computes the Euclidean (L2) distance between rows of X and Y.
# Uses torch.cdist with p=2 to efficiently compute pairwise distances between vectors using the Euclidean norm.
# Here, the parameter 'p' specifies the order of the norm, with p=2 indicating the L2 (Euclidean) norm.
def distance_l2(X, Y):
    return torch.cdist(X, Y, p=2)

# Computes a distance measure based on the negative dot product between rows of X and Y.
# In many similarity tasks, a higher dot product indicates greater similarity. Here we use the negative value to convert it into a distance metric.
# This is computed efficiently using torch.matmul.
def distance_dot(X, Y):
    return -torch.matmul(X, Y.t())

# Computes the Manhattan (L1) distance between rows of X and Y.
# torch.cdist with p=1 is utilized to compute the sum of absolute differences (Manhattan distance) between vectors.
# Here, the 'p' parameter specifies the order of the norm, with p=1 indicating the L1 (Manhattan) norm.
def distance_manhattan(X, Y):
    return torch.cdist(X, Y, p=1)

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# Implements k-nearest neighbor (KNN) search using PyTorch optimized operations.
def our_knn(N, D, A, X, K):
    # Compute pairwise Euclidean distances between each query point in X and all dataset points in A using the L2 (Euclidean) norm
    distances = distance_l2(X, A)

    # For each query point, obtain the indices of the K smallest distances (nearest neighbors) using torch.topk with largest=False
    _, indices = torch.topk(distances, k=K, dim=1, largest=False)

    return indices

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# Implements k-means clustering using PyTorch optimized operations.
def our_kmeans(N, D, A, K):
    # Initialize centroids by randomly selecting K data points from A
    centroids = A[torch.randperm(N)[:K]].clone()
    max_iter = 100
    for i in range(max_iter):
         # Compute Euclidean distances between each data point in A and the centroids using our distance_l2 function
         distances = distance_l2(A, centroids)

         # Assign each data point to the nearest centroid
         assignments = torch.argmin(distances, dim=1)

         # Compute new centroids as the sum of points in each cluster using vectorized index_add
         new_centroids = torch.zeros_like(centroids)
         new_centroids = new_centroids.index_add(0, assignments, A)

         # Count the number of points assigned to each centroid
         counts = torch.bincount(assignments, minlength=K).view(K, 1)

         # For any cluster with zero points, reinitialize its centroid randomly
         zero_mask = (counts == 0).view(-1)
         if zero_mask.any():
             new_centroids[zero_mask] = A[torch.randint(0, N, (int(zero_mask.sum().item()),))]
             counts[zero_mask] = 1

         # Update centroids as the mean of assigned points
         new_centroids = new_centroids / counts.float()

         # Check for convergence: if centroids do not change significantly, break
         if torch.allclose(new_centroids, centroids, atol=1e-4):
             break
         centroids = new_centroids
    return centroids

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# Implements approximate nearest neighbor (ANN) search using k-means clustering.
def our_ann(N, D, A, X, K):
    # Choose a number of clusters for the ANN approximation. This is a trade-off between speed and accuracy.
    num_clusters = min(50, max(1, N // 10))
    
    # Cluster the dataset A using our_kmeans into 'num_clusters' clusters
    centroids = our_kmeans(N, D, A, num_clusters)
    
    # Assign each data point in A to the nearest centroid
    distances_A = distance_l2(A, centroids)  # shape: (N, num_clusters)
    assignments_A = torch.argmin(distances_A, dim=1)  

    # For each query in X, assign it to the nearest centroid
    distances_X = distance_l2(X, centroids)  # shape: (M, num_clusters), M = number of query points
    query_assignments = torch.argmin(distances_X, dim=1)  

    M = X.size(0)
    ann_indices = []
    for i in range(M):
        # Get the cluster index for the i-th query
        cluster_idx = query_assignments[i].item()
        
        # Find the indices in A that belong to this cluster
        cluster_mask = (assignments_A == cluster_idx)
        indices_in_cluster = cluster_mask.nonzero(as_tuple=False).squeeze()
        
        # If no points are found in this cluster or if there are fewer than K points, fallback to global search
        if indices_in_cluster.numel() == 0 or indices_in_cluster.numel() < K:
            d_full = distance_l2(X[i].unsqueeze(0), A)
            _, idx_full = torch.topk(d_full, k=K, dim=1, largest=False)
            ann_indices.append(idx_full.squeeze(0))
        else:
            # Retrieve the subset of A corresponding to the selected cluster
            A_cluster = A[indices_in_cluster]
            # Compute distances from the query to points in this cluster
            d_cluster = distance_l2(X[i].unsqueeze(0), A_cluster)
            # If the cluster has fewer points than K, fallback to global search
            if d_cluster.size(1) < K:
                d_full = distance_l2(X[i].unsqueeze(0), A)
                _, idx_full = torch.topk(d_full, k=K, dim=1, largest=False)
                ann_indices.append(idx_full.squeeze(0))
            else:
                # Find the K nearest neighbors within the cluster
                _, local_idx = torch.topk(d_cluster, k=K, dim=1, largest=False)
                # Map the local indices back to the indices in the full dataset A
                selected_indices = indices_in_cluster[local_idx.squeeze(0)]
                ann_indices.append(selected_indices)

    # Stack results into a tensor of shape (M, K) and return
    return torch.stack(ann_indices, dim=0)

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
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
    test_kmeans()
