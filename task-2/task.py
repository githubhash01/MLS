# import torch
import cupy as cp
import cupyx
# import triton
import numpy as np
import time
import json
# from test import testdata_kmeans, testdata_knn, testdata_ann
# from sklearn.datasets import make_blobs

from distance_functions import (
        distance_l2_gpu, 
        distance_cosine_gpu,
        distance_dot_gpu,
        distance_manhattan_gpu,
        distance_l2_cpu, 
        distance_cosine_cpu,
        distance_dot_cpu,
        distance_manhattan_cpu,
        distance_l2_kmeans,
        distance_cosine_kmeans,
        distance_manhattan_kmeans,
        distance_dot_kmeans
    )
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# def distance_cosine(X, Y):
#     return 1 - (cp.dot(X, Y) / (cp.linalg.norm(X) * cp.linalg.norm(Y)))

# def distance_l2(X, Y):
#     return cp.sqrt(cp.sum((X - Y) ** 2))

# def distance_dot(X, Y):
#     return cp.dot(X, Y)

# def distance_manhattan(X, Y):
#     return cp.sum(cp.absolute(X - Y))

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

def our_knn_cupy(N, D, A, X, K, distance_fn=distance_l2_gpu):
    """
    Input:
        N: Number of vectors
        D: Dimension of vectors
        A[N, D]: A collection of vectors (CuPy array)
        X[D]: A query vector (CuPy array)
        K: Top K
        distance_fn: A vectorized function that computes distances between X and all rows in A

    Output:
        sorted_k_indices[K]: Indices of nearest neighbors
        k_distances[K]: Corresponding distances
    """
    X = X.reshape(1, D)
    distances = distance_fn(A, X)

    top_k_indices = cp.argpartition(distances, K)[:K]
    sorted_k_indices = top_k_indices[cp.argsort(distances[top_k_indices])]
    k_distances = distances[sorted_k_indices]

    return sorted_k_indices, k_distances

def our_knn_raw_tiled(N, D, A, X, K):
    knn_kernel_tiled = cp.RawKernel(r'''
    extern "C" __global__ void knn_kernel_tiled(const double* A, const double* X, double* distances, int* indices, int N, int D, int tile_size) {
        extern __shared__ double shared_A[];  // Shared memory for a tile of A
        int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index
        int tid = threadIdx.x;  // Thread index within the block

        double dist = 0.0;

        // Process A in tiles
        for (int tile_start = 0; tile_start < D; tile_start += tile_size) {
            // Load a tile of A into shared memory
            int tile_end = min(tile_start + tile_size, D);
            for (int j = tile_start + tid; j < tile_end; j += blockDim.x) {
                shared_A[j - tile_start] = A[idx * D + j];
            }
            __syncthreads();  // Synchronize threads to ensure shared_A is fully populated

            // Compute partial distance for the current tile
            for (int j = 0; j < tile_end - tile_start; j++) {
                double diff = shared_A[j] - X[tile_start + j];
                dist += diff * diff;
            }
            __syncthreads();  // Synchronize threads before loading the next tile
        }

        // Write the final distance and index
        if (idx < N) {
            distances[idx] = dist;
            indices[idx] = idx;
        }
    }
    ''', 'knn_kernel_tiled')

    distances = cp.zeros(N, dtype='float64')   # Output distances
    indices = cp.zeros(N, dtype='int32')       # Output indices

    # Define the block size and grid size
    block_size = 256  # Example block size
    grid_size = (N + block_size - 1) // block_size  # Ensure all elements are covered

    # Define the tile size (must fit within shared memory)
    shared_mem_limit = cp.cuda.Device().attributes['MaxSharedMemoryPerBlock']
    tile_size = shared_mem_limit // cp.dtype('float64').itemsize  # Max elements per tile
    print("shmem limit", shared_mem_limit)
    print(f"Tile size: {tile_size} elements")

    # Launch the kernel
    knn_kernel_tiled(
        (grid_size,),  # Grid dimensions (1D grid)
        (block_size,),  # Block dimensions (1D block)
        (A, X, distances, indices, N, D, tile_size),  # Kernel arguments
        shared_mem=tile_size * cp.dtype('float64').itemsize  # Shared memory size
    )

    K = min(K, N)
    top_k_indices = cp.argpartition(distances, K)[:K]
    sorted_k_indices = top_k_indices[cp.argsort(distances[top_k_indices])]

    return indices[sorted_k_indices], distances[sorted_k_indices]

def our_knn_cpu(N, D, A, X, K, distance_func=distance_l2_cpu):
    """
    Input:
        N: Number of vectors
        D: Dimension of vectors
        A[N, D]: A collection of vectors (NumPy array)
        X: A specified vector (NumPy array of shape [D])
        K: Top K (number of nearest neighbors to find)
        distance_func: Distance function to use (default: L2)
    
    Output:
        indices: The indices of the K-nearest neighbors in A
        distances: The corresponding distances of the K-nearest neighbors
    """
    X = X.reshape(D)  # Ensure X is 1D (original functions expect 1D inputs)

    distances = distance_func(A, X)

    top_k_indices = np.argpartition(distances, K)[:K]
    sorted_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]
    k_distances = distances[sorted_k_indices]

    return sorted_k_indices, k_distances


# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def kmeans_plus_plus_init(A, K, distance_fn=distance_l2_kmeans):
    """
    K-means++ initialization to select K centroids for clustering.
    
    Input:
        A: A collection of vectors [N, D]
        K: Number of centroids to choose
        distance_fn: Distance function to compute distance (default: L2 distance)

    Output:
        centroids: The initialized centroids [K, D]
    """
    N = A.shape[0]
    
    # Choose the first centroid randomly
    centroids = cp.zeros((K, A.shape[1]), dtype=A.dtype)
    centroids[0] = A[cp.random.randint(0, N)]
    
    # Initialize an array to store distances
    distances = cp.full(N, cp.inf, dtype=A.dtype)
    
    for k in range(1, K):
        # Calculate distance from each point to the closest centroid
        dist_to_centroid = distance_fn(A, centroids[:k, :])  # [N, k]
        min_dist_to_centroid = cp.min(dist_to_centroid)  # [N]
        
        # Update distances
        distances = cp.minimum(distances, min_dist_to_centroid)
        
        # Choose the next centroid with probability proportional to distance^2
        prob = distances ** 2
        
        # Ensure that all values in prob are non-negative (they should be, but let's check just in case)
        prob = cp.maximum(prob, 0)  # Set any negative values to 0
        
        # Handle potential cases where sum(prob) is zero (e.g., all points are identical or very close)
        if cp.sum(prob) == 0:
            prob = cp.ones(N)  # If all probabilities are zero, equally select any point
        
        prob /= cp.sum(prob)  # Normalize to make it a probability distribution
        
        # Choose the next centroid based on the probability distribution
        chosen_idx = cp.random.choice(N, size=1, p=prob)
        centroids[k] = A[chosen_idx]
    
    return centroids

def our_kmeans(N, D, A, K, distance_fn=distance_l2_gpu, centroid_distance_fn=distance_l2_kmeans, max_iters=100, tol=1e-5):
    """
    Input:
        N: Number of vectors
        D: Dimension of vectors
        A[N, D]: A collection of vectors (CuPy array)
        K: Number of clusters
        distance_fn: Which distance function
        max_iters: Maximum number of iterations
        tol: Convergence tolerance

    Output:
        centroids[K, D]: Final cluster centroids
        labels[N]: Cluster assignments for each vector
    """
    # Initialize centroids from random points in A
    # random_indices = cp.random.choice(N, K, replace=False)
    # centroids = A[random_indices]

    centroids = kmeans_plus_plus_init(A, K, centroid_distance_fn)

    for _ in range(max_iters):
        # Compute distances using provided distance function
        centroid_distances = centroid_distance_fn(A, centroids)  # [N, K]

        # Assign clusters
        labels = cp.argmin(centroid_distances, axis=1)

        # Recompute centroids
        new_centroids = cp.zeros((K, D), dtype=A.dtype)
        for k in range(K):
            cluster_points = A[labels == k]
            if cluster_points.shape[0] > 0:
                new_centroids[k] = cp.mean(cluster_points, axis=0)
            else:
                new_centroids[k] = A[cp.random.randint(0, N)]

        # Convergence check
        if cp.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return centroids, labels


def numpy_kmeans(N, D, A, K, max_iters=150, tol=1e-5):
    """
    K-means clustering using NumPy (CPU).
    
    Input:
        N: Number of vectors
        D: Dimension of vectors
        A[N, D]: A collection of vectors (NumPy array)
        K: Number of clusters
    Output:
        centroids[K, D]: Final cluster centroids
        labels[N]: Cluster assignments for each vector
    """
    # Randomly initialize centroids by selecting K unique points from A
    random_indices = np.random.choice(N, K, replace=False)
    centroids = A[random_indices]

    for i in range(max_iters):
        # Compute distances between each point and each centroid
        # Resulting shape: [N, K]
        distances = np.linalg.norm(A[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)

        # Assign each point to the closest centroid
        labels = np.argmin(distances, axis=1)

        # Compute new centroids
        new_centroids = np.zeros((K, D), dtype=A.dtype)
        for k in range(K):
            cluster_points = A[labels == k]
            if cluster_points.shape[0] > 0:
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Reinitialize empty clusters
                new_centroids[k] = A[np.random.randint(0, N)]

        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return centroids, labels

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K, centroids, labels, distance_fn=distance_l2_gpu, centroid_distance_fn=distance_l2_kmeans, num_clusters=100):
    """
    Approximate KNN using K-means clustering with selectable distance functions (on GPU with CuPy).

    Input:
        N: Number of vectors
        D: Dimension of vectors
        A[N, D]: Collection of vectors (CuPy array)
        X[D]: Query vector (CuPy array)
        K: Top K nearest neighbors to return
        centroids[K, D]: Precomputed centroids from K-means
        labels[N]: Precomputed labels (cluster assignments) from K-means
        distance_fn: Function to compute distances from X to candidates (vectorized, returns [C])
        num_clusters: Total number of clusters

    Output:
        indices[K]: Indices of approximate nearest neighbors
        distances[K]: Distances to the nearest neighbors
    """
    X = X.reshape(1, D)

    # Find closest centroid
    centroid_distances = centroid_distance_fn(centroids, X)  # [K]
    closest_cluster = cp.argmin(centroid_distances)
    print(centroid_distances)
    # Select candidate points from that cluster
    cluster_indices = cp.where(labels == closest_cluster)[0]
    candidates = A[cluster_indices]
    print(candidates)

    if candidates.shape[0] == 0:
        # Fallback to full dataset if cluster is empty
        cluster_indices = cp.arange(N)
        candidates = A

    # Compute distances to candidates and select top K
    distances = distance_fn(candidates, X)  # [C]
    top_k_idx = cp.argsort(distances)[:K]

    return cluster_indices[top_k_idx], distances[top_k_idx]

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example

def measure_speedup_knn(N, D, K):
    # Generate random data
    A_gpu = cp.random.rand(N, D)  # CuPy array for GPU
    X_gpu = cp.random.rand(D)     # CuPy vector for GPU

    A_cpu = np.random.rand(N, D)  # NumPy array for CPU
    X_cpu = np.random.rand(D)     # NumPy vector for CPU

    our_knn_cupy(N, D, A_gpu, X_gpu, K)

    # Measure time for the GPU implementation
    start_gpu = time.perf_counter()
    indices_gpu, distances_gpu = our_knn_cupy(N, D, A_gpu, X_gpu, K)
    end_gpu = time.perf_counter()
    gpu_time = end_gpu - start_gpu

    our_knn_cpu(N, D, A_gpu, X_gpu, K)

    # Measure time for the CPU implementation
    start_cpu = time.perf_counter()
    indices_cpu, distances_cpu = our_knn_cpu(N, D, A_cpu, X_cpu, K)
    end_cpu = time.perf_counter()
    cpu_time = end_cpu - start_cpu

    # Calculate and print the speedup
    speedup = cpu_time / gpu_time
    print(f"CPU Time = {cpu_time:.6f}s, GPU Time = {gpu_time:.6f}s, Speedup = {speedup:.2f}x")
    # assert cp.allclose(indices_gpu, indices_gpu, atol=1e-6), "Mismatch in results!"

def measure_speedup_kmeans(N, D, K):
    # Generate random data
    cp.random.seed(12345)
    A_gpu = cp.random.rand(N, D)  # CuPy array for GPU
    # X_gpu = cp.random.rand(D)     # CuPy vector for GPU

    np.random.seed(12345)
    A_cpu = np.random.rand(N, D)  # NumPy array for CPU
    # X_cpu = np.random.rand(D)     # NumPy vector for CPU

    our_kmeans(N, D, A_gpu, K)

    # Measure time for the GPU implementation
    start_gpu = time.perf_counter()
    cupy_centroids, cupy_labels = our_kmeans(N, D, A_gpu, K)
    # print(cupy_centroids, cupy_labels)
    end_gpu = time.perf_counter()
    gpu_time = end_gpu - start_gpu

    numpy_kmeans(N, D, A_cpu, K)

    # Measure time for the CPU implementation
    start_cpu = time.perf_counter()
    cpu_centroids, cpu_labels = numpy_kmeans(N, D, A_cpu, K)
    # print(cpu_centroids, cpu_labels)
    end_cpu = time.perf_counter()
    cpu_time = end_cpu - start_cpu

    # Calculate and print the speedup
    speedup = cpu_time / gpu_time
    print(f"CPU Time = {cpu_time:.6f}s, GPU Time = {gpu_time:.6f}s, Speedup = {speedup:.2f}x")


def benchmark_knn(func, N, D, A, X, K, runs=5):
    # Warm-up run (to avoid startup overhead

    ind, _ = func(N, D, A, X, K)
    print(ind)

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    start_event.record()
    for _ in range(runs):
        func(N, D, A, X, K)
    end_event.record()

    # Wait for GPU to finish (synchronize)
    end_event.synchronize()
    
    elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event) / runs  # ms
    return elapsed_time

def new_benchmark_knn(N, D, K):
    cp.random.seed(12345)

    A = cp.random.rand(N, D)  # CuPy array for GPU
    X = cp.random.rand(D)     # CuPy vector for GPU
    # Measure performance
    time_cupy = benchmark_knn(our_knn_cupy, N, D, A, X, K)
    time_raw = benchmark_knn(our_knn_cpu, N, D, A, X, K)

    print(f"Pure CuPy Time: {time_cupy:.4f} ms")
    print(f"CPU Kernel Time: {time_raw:.4f} ms")
    print(f"Speedup: {time_cupy / time_raw:.2f}x")

def new_benchmark_kmeans(N, D, K):
    cp.random.seed(12345)

    A = cp.random.rand(N, D)  # CuPy array for GPU
    X = cp.random.rand(D)     # CuPy vector for GPU
    # Measure performance
    time_cupy = benchmark_knn(our_knn_cupy, N, D, A, X, K)
    time_raw = benchmark_knn(our_knn_cpu, N, D, A, X, K)

    print(f"Pure CuPy Time: {time_cupy:.4f} ms")
    print(f"CPU Kernel Time: {time_raw:.4f} ms")
    print(f"Speedup: {time_cupy / time_raw:.2f}x")

def profile_gpu_knn(func, N, D, K):
    cp.random.seed(12345)

    A = cp.random.rand(N, D)  # CuPy array for GPU
    X = cp.random.rand(D)     # CuPy vector for GPU

    indices, _ = func(N, D, A, X, K)
    print(indices)

    return

def compare_ann_recall_with_cupy(N, D, A_cpu, queries_cpu, K, num_clusters):
    """
    Compare ANN recall against exact KNN computed with CuPy.

    Input:
        N: Number of vectors
        D: Dimension of vectors
        A_cpu[N, D]: Dataset vectors (NumPy array)
        queries_cpu[Q, D]: Query vectors (NumPy array)
        K: Top K neighbors to retrieve
        num_clusters: Number of clusters used in ANN

    Output:
        avg_recall: Average recall@K across all queries
    """
    A_gpu = cp.asarray(A_cpu)
    Q = queries_cpu.shape[0]
    total_recall = 0.0

    centroids, labels = our_kmeans(N, D, A_gpu, num_clusters, distance_fn=distance_cosine_gpu, centroid_distance_fn=distance_cosine_kmeans)

    for i in range(Q):
        query_cpu = queries_cpu[i]
        query_gpu = cp.asarray(query_cpu)

        # Exact KNN on GPU
        true_indices, _ = our_knn_cupy(N, D, A_gpu, query_gpu, K, distance_fn=distance_cosine_gpu)
        true_set = set(cp.asnumpy(true_indices))

        # ANN on CPU
        approx_indices, _ = our_ann(N, D, A_gpu, query_gpu, K, centroids, labels, distance_cosine_gpu, distance_cosine_kmeans, num_clusters)
        approx_set = set(cp.asnumpy(approx_indices))

        # Compute recall@K
        hits = len(approx_set.intersection(true_set))
        total_recall += hits / K

        print("rec: ", len(approx_set & true_set) / K)

    avg_recall = total_recall / Q
    print(f"Recall@{K} over {Q} queries: {avg_recall:.4f}")
    return avg_recall

# Run the speedup measurements
# if __name__ == "__main__":
#     # test_distances()
#     # measure_speedup_knn(2**15, 20, 5)
#     # measure_speedup_kmeans(2**10, 2**10, 3)
#     # Generate data
#     # cp.random.seed(12345)
#     N, D = 2**10, 64
#     # A = cp.random.rand(N, D).astype(cp.float32)
#     # queries = cp.random.rand(100, D).astype(cp.float32)
#     K = 5
#     n_clusters = 20

#     A, _ = make_blobs(n_samples=N, n_features=D, centers=n_clusters, random_state=12345)
#     A = cp.array(A, dtype=cp.float32)
#     min_A = cp.min(A, axis=0)
#     max_A = cp.max(A, axis=0)

#     # Generate random queries in the same feature space (same order of magnitude)
#     queries = cp.random.rand(100, D).astype(cp.float32)
#     # Scale queries to be within the range of A
#     queries = min_A + (max_A - min_A) * queries

#     compare_ann_recall_with_cupy(N, D, A, queries, K, num_clusters=n_clusters)
#     # profile_gpu_knn(our_knn_cupy, 20, 2**20, 10)
#     # profile_gpu_knn(our_knn_raw_tiled, 2**15, 20, 10)

#     # profile_knn(our_knn_raw, 100000, 10, 5)
#     # new_benchmark_knn(2**15, 20, 5)

