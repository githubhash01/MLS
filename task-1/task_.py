import numpy as np
import cupy as cp
import time
import json
# from distance_functions import (
#         distance_l2_gpu, 
#         distance_cosine_gpu,
#         distance_dot_gpu,
#         distance_manhattan_gpu,
#         distance_l2_cpu, 
#         distance_cosine_cpu,
#         distance_dot_cpu,
#         distance_manhattan_cpu,
#         distance_l2_kmeans,
#         distance_cosine_kmeans,
#         distance_manhattan_kmeans,
#         distance_dot_kmeans
#     )

# # ------------------------------------------------------------------------------------------------
# # CPU 1.1a - Distance functions
# # ------------------------------------------------------------------------------------------------

def distance_l2_cpu(A, X):
    return np.linalg.norm(A - X, axis=1)


def distance_cosine_cpu(A, X):
    A_norm = np.linalg.norm(A, axis=1)
    X_norm = np.linalg.norm(X)
    dot = A @ X
    return 1 - (dot / (A_norm * X_norm + 1e-8))  # Avoid divide by 0


def distance_manhattan_cpu(A, X):
    return np.sum(np.abs(A - X), axis=1)


def distance_dot_cpu(A, X):
    return A @ X


# # ----------------------------------------------------------------------------------
# #  CPU 1.1b - Distance functions for KMeans
# # ----------------------------------------------------------------------------------
dist_kernel = cp.RawKernel(r'''
extern "C" _global_ void calc_distances(const float *data,
                                            const float *centers,
                                            int N, int K, int D,
                                            float *dist)
{
    int point = blockDim.x * blockIdx.x + threadIdx.x;
    int cluster = blockDim.y * blockIdx.y + threadIdx.y;

    if (point < N && cluster < K) {
        float sum = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = data[point * D + d] - centers[cluster * D + d];
            sum += diff * diff;
        }
        dist[point * K + cluster] = sqrtf(sum);
    }
}
''', 'calc_distances')


def distance_l2_kmeans_kernel(A, C):
    """
    Compute the pairwise squared L2 distances between points in A and centers in C
    using a modified custom CUDA kernel that is optimized for high dimension.

    Parameters:
      A : cp.ndarray, shape (N, D) - Data points.
      C : cp.ndarray, shape (K, D) - Cluster centers.

    Returns:
      cp.ndarray of shape (N, K) with the squared L2 distances.
    """
    # Convert input arrays to float32 explicitly.
    A = cp.asarray(A, dtype=cp.float32)
    C = cp.asarray(C, dtype=cp.float32)
    N, D = A.shape
    K = C.shape[0]

    # Allocate output array for distances.
    dist = cp.zeros((N, K), dtype=cp.float32)

    # Define block and grid dimensions. Tune block size based on your GPU.
    block_x = 8
    block_y = 8
    grid_x = (N + block_x - 1) // block_x
    grid_y = (K + block_y - 1) // block_y
    grid = (grid_x, grid_y)
    block = (block_x, block_y)

    # Launch the kernel.
    dist_kernel(grid, block, (A, C, N, K, D, dist))

    return dist


def distance_l2_kmeans_cpu(A, C):
    # A: [N, D], C: [K, D]
    return np.linalg.norm(A[:, np.newaxis, :] - C[np.newaxis, :, :], axis=2)


def distance_cosine_kmeans_cpu(A, C):
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)  # Shape [N, 1]
    C_norm = np.linalg.norm(C, axis=1, keepdims=True)  # Shape [K, 1]

    sim = np.dot(A, C.T) / (A_norm * C_norm.T + 1e-8)  # Shape [N, K]

    return 1 - sim  # Cosine distance, shape [N, K]


def distance_dot_kmeans_cpu(A, C):
    """Dot similarity (negated for distance)"""
    return -A @ C.T  # Higher dot product = closer


def distance_manhattan_kmeans_cpu(A, C):
    """Manhattan (L1) distance between A and centroids"""
    return np.sum(np.abs(A[:, np.newaxis, :] - C[np.newaxis, :, :]), axis=2)


# # ------------------------------------------------------------------------------------------------
# # GPU 1.1a - Distance functions
# # ------------------------------------------------------------------------------------------------
def distance_l2_gpu(X, Y):
    return cp.linalg.norm(X - Y, axis=1)


def distance_cosine_gpu(A, X):
    if X.ndim == 1:  # If X is a single vector, reshape it to [1, D]
        X = X[cp.newaxis, :]  # Shape becomes [1, D]

    A_norm = cp.linalg.norm(A, axis=1)  # [N]
    X_norm = cp.linalg.norm(X)  # scalar
    dot = cp.dot(A, X.T)  # [N, 1], dot product between A and X

    return 1 - (dot.flatten() / (A_norm * X_norm + 1e-8))


def distance_manhattan_gpu(X, Y):
    return cp.sum(cp.abs(X - Y), axis=1)


def distance_dot_gpu(X, Y):
    return cp.dot(X, Y)


# ----------------------------------------------------------------------------------
# GPU 1.1b - Distance functions for KMeans
# ----------------------------------------------------------------------------------

def distance_l2_kmeans_gpu(A, C):
    # A: [N, D], C: [K, D]
    return cp.linalg.norm(A[:, cp.newaxis, :] - C[cp.newaxis, :, :], axis=2)


def distance_cosine_kmeans_gpu(A, C):
    A_norm = cp.linalg.norm(A, axis=1, keepdims=True)  # Shape [N, 1]
    C_norm = cp.linalg.norm(C, axis=1, keepdims=True)  # Shape [K, 1]

    sim = cp.dot(A, C.T) / (A_norm * C_norm.T + 1e-8)  # Shape [N, K]

    return 1 - sim  # Cosine distance, shape [N, K]


def distance_dot_kmeans_gpu(A, C):
    return -A @ C.T  # Higher dot product = closer


def distance_manhattan_kmeans_gpu(A, C):
    return cp.sum(cp.abs(A[:, cp.newaxis, :] - C[cp.newaxis, :, :]), axis=2)


# ------------------------------------------------------------------------------------------------
# CPU 1.2 - K Nearest Neighbors
# ------------------------------------------------------------------------------------------------

def our_knn_cpu(N, D, A, X, K, distance_func):
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
# GPU 1.2 - K Nearest Neighbors
# ------------------------------------------------------------------------------------------------

def our_knn_gpu(N, D, A, X, K, distance_fn):
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

    distances = cp.zeros(N, dtype='float64')  # Output distances
    indices = cp.zeros(N, dtype='int32')  # Output indices

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


# ------------------------------------------------------------------------------------------------
# CPU 2.1 - KMeans
# ------------------------------------------------------------------------------------------------

# Default - sklearn (CPU)
# from sklearn.cluster import KMeans


# def kmeans_sklearn(N, D, A, K):
#     kmeans = KMeans(n_clusters=K)
#     kmeans.fit(A)
#     return kmeans.cluster_centers_, kmeans.labels_


def kmeans_pp(N, D, A, K):
    """
    Initialize centroids using the k-means++ algorithm with NumPy.
    """
    centroids = []

    # 1. Randomly choose the first centroid.
    first_idx = np.random.choice(N)
    centroids.append(A[first_idx])

    # 2. Select the remaining centroids.
    for _ in range(1, K):
        centroids_array = np.vstack(centroids)  # shape: (current_num_centroids, D)
        distances = np.linalg.norm(A[:, None, :] - centroids_array[None, :, :], axis=2)
        min_dists = np.min(distances, axis=1)

        squared_dists = min_dists ** 2
        total = np.sum(squared_dists)

        if total == 0:
            next_idx = np.random.choice(N)
        else:
            probs = squared_dists / total
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()
            next_idx = np.searchsorted(cumulative_probs, r)

        centroids.append(A[next_idx])

    return np.array(centroids)


# Favoured for its simplicity and speed
def maximin_cpu(N, D, A, K):
    # 1) Randomly choose the first center.
    first_idx = np.random.choice(N)
    centers = A[[first_idx]]  # centers now has shape (1, D)

    # 2) Choose each subsequent centroid
    for _ in range(1, K):
        # Calculate the distances from every point to every center.
        # distances will have shape (N, current_num_centers)
        distances = np.linalg.norm(A[:, None, :] - centers[None, :, :], axis=2)

        # For each data point, find its minimum distance to any center.
        min_dists = np.min(distances, axis=1)  # shape: (N,)

        # The next center is the one with the maximum distance among these minimum distances.
        next_idx = np.argmax(min_dists)

        # Append the new center and reassign.
        centers = np.vstack([centers, A[next_idx]])

    return centers


def our_kmeans_cpu(N, D, A, K, max_iter=100):
    centers = maximin_cpu(N, D, A, K)
    assignments = np.zeros(N)  # initialise assignments all to 0

    for _ in range(max_iter):

        # 1) Calculate distance of each point to centers
        distances = np.linalg.norm(A[:, None, :] - centers[None, :, ], axis=2)

        # 2) Assign each point to the closest center
        new_assignments = np.argmin(distances, axis=1)

        # 3) Check for convergence
        if np.array_equal(assignments, new_assignments):
            break

        # 4) Calculate new centers based on new assignments
        for cluster_idx in range(K):
            centers[cluster_idx] = A[new_assignments == cluster_idx].mean(axis=0)

        # 5) Update assignments
        assignments = new_assignments

    return centers, assignments


# ------------------------------------------------------------------------------------------------
# GPU 2.1 - KMeans
# ------------------------------------------------------------------------------------------------
# A must be on GPU
def maximin_cupy(N, D, A, K, dist_func):
    first_idx = cp.random.choice(N, 1).item()
    centers = A[first_idx:first_idx + 1]

    for _ in range(1, K):
        distances = dist_func(A, centers)
        min_dists = cp.min(distances, axis=1)
        next_idx = int(cp.argmax(min_dists))
        centers = cp.vstack([centers, A[next_idx:next_idx + 1]])

    return centers


# A must be GPU
def update_centers_cupy(N, D, A, K, assignments):
    sums = cp.zeros((K, D), dtype=A.dtype)
    cp.add.at(sums, assignments, A)
    counts = cp.bincount(assignments, minlength=K)
    return sums / counts[:, None]


def kmeans_gpu(N, D, A, K, max_iter=100, dist_func=distance_l2_kmeans_gpu):
    # cast numpy A to cupy (GPU)
    A_gpu = cp.asarray(A)
    centers = maximin_cupy(N, D, A_gpu, K, dist_func)
    assignments = cp.zeros(N)

    for it in range(max_iter):

        # 1) Calculate distance of each point to centers
        distances = dist_func(A_gpu, centers)

        # 2) Assign each point to closest center
        new_assignments = cp.argmin(distances, axis=1)

        # 3) Check for convergence
        if cp.array_equal(assignments, new_assignments):
            break

        # 4) Calculate new centers based on new assignments
        centers = update_centers_cupy(N, D, A, K, new_assignments)

        # 5) Update assignments
        assignments = new_assignments

    # Convert GPU array if needed
    # centers = cp.asnumpy(centers)
    # assignments = cp.asnumpy(assignments)
    return centers, assignments


def maximin_batched(N, D, A, K, dist_func, batch_size):
    # Pick a random point for the first center.
    first_idx = int(cp.random.choice(N, 1).item())
    centers = A[first_idx:first_idx + 1].copy()  # shape (1, D)

    for i in range(1, K):
        # For each point in A_gpu, compute its distance to the closest center (batched).
        min_dists = cp.empty(N, dtype=cp.float32)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            # Compute distances between this batch and all current centers.
            # This returns shape (batch_size, i)
            dists_batch = dist_func(A[start:end], centers)
            # For each point in the batch, take the minimum distance among the centers.
            min_dists[start:end] = cp.min(dists_batch, axis=1)

        # Choose the point with the maximum minimum distance.
        next_idx = int(cp.argmax(min_dists).item())
        # Append that point to centers.
        centers = cp.concatenate([centers, A[next_idx:next_idx + 1]], axis=0)

    return centers


def kmeans_gpu_batched(N, D, A, K, max_iter=100, batch_size=16_000, dist_func=distance_l2_kmeans_kernel): #, centroid_distance_fn=distance_l2_kmeans_kernel):
    # Upload full dataset onto GPU once
    A_gpu = cp.asarray(A, dtype=cp.float32)
    assignments = cp.zeros(N, dtype=cp.int32)
    new_assignments = cp.empty(N, dtype=cp.int32)

    # Get initial centers using a batched maximin method.
    centers = maximin_batched(N, D, A_gpu, K, dist_func, batch_size)

    # Precompute batch indices and create streams for each batch
    batch_indices = [(start, min(start + batch_size, N)) for start in range(0, N, batch_size)]
    streams = [cp.cuda.Stream() for _ in batch_indices]

    for it in range(max_iter):

        # Calculate distances and find closest centers
        for (start, end), stream in zip(batch_indices, streams):
            with stream:
                # 1) Calculate distance of each point to centers for current batch
                distances_batch = dist_func(A_gpu[start:end], centers)

                # 2) Assign each point to closest center in the current batch.
                new_assignments[start:end] = cp.argmin(distances_batch, axis=1)

        # synchronise all streams
        for stream in streams:
            stream.synchronize()

        # 3) Check for convergence
        if cp.array_equal(assignments, new_assignments):
            break

        # 4) Calculate new centers based on new assignments
        centers = update_centers_cupy(N, D, A_gpu, K, new_assignments)

        # 5) Update assignments
        assignments[:] = new_assignments

    return centers, assignments


# ------------------------------------------------------------------------------------------------
# GPU 2.2 - Approximate KNN using KMeans
# ------------------------------------------------------------------------------------------------

def our_ann(N, D, A, X, K, centroids, labels, distance_fn=distance_l2_gpu, centroid_distance_fn=distance_l2_kmeans_gpu,
            num_clusters=100):
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

    # Select candidate points from that cluster
    cluster_indices = cp.where(labels == closest_cluster)[0]
    candidates = A[cluster_indices]

    if candidates.shape[0] == 0:
        # Fallback to full dataset if cluster is empty
        cluster_indices = cp.arange(N)
        candidates = A

    # Compute distances to candidates and select top K
    distances = distance_fn(candidates, X)  # [C]
    top_k_idx = cp.argsort(distances)[:K]

    return cluster_indices[top_k_idx], distances[top_k_idx]


def our_ann_cpu(N, D, A, X, K, centroids, labels, distance_fn=distance_l2_cpu, centroid_distance_fn=distance_l2_kmeans_cpu,
            num_clusters=100):
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
    closest_cluster = np.argmin(centroid_distances)

    # Select candidate points from that cluster
    cluster_indices = np.where(labels == closest_cluster)[0]
    candidates = A[cluster_indices]

    if candidates.shape[0] == 0:
        # Fallback to full dataset if cluster is empty
        cluster_indices = np.arange(N)
        candidates = A

    # Compute distances to candidates and select top K
    distances = distance_fn(candidates, X)  # [C]
    top_k_idx = np.argsort(distances)[:K]

    return cluster_indices[top_k_idx], distances[top_k_idx]

def compare_ann_recall_with_cupy(N, D, A_cpu, queries_cpu, K, num_clusters, batch_size):
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

    centroids, labels = kmeans_gpu_batched(N, D, A_gpu, num_clusters, batch_size=batch_size, dist_func=distance_cosine_gpu, centroid_distance_fn=distance_cosine_kmeans_gpu)

    for i in range(Q):
        query_cpu = queries_cpu[i]
        query_gpu = cp.asarray(query_cpu)

        # Exact KNN on GPU
        true_indices, _ = our_knn_gpu(N, D, A_gpu, query_gpu, K, distance_fn=distance_cosine_gpu)
        true_set = set(cp.asnumpy(true_indices))

        # ANN on CPU
        approx_indices, _ = our_ann(N, D, A_gpu, query_gpu, K, centroids, labels, distance_cosine_gpu, distance_cosine_kmeans_gpu, num_clusters)
        approx_set = set(cp.asnumpy(approx_indices))

        # Compute recall@K
        hits = len(approx_set.intersection(true_set))
        total_recall += hits / K

        # print("rec: ", len(approx_set & true_set) / K)

    avg_recall = total_recall / Q
    print(f"Recall@{K} over {Q} queries: {avg_recall:.4f}")
    return avg_recall

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------


if __name__=="__main__":
    import csv 
    def benchmark(func_cpu, func_gpu, dimensions, N=4096):
        cpu_times = []
        gpu_times = []

        for D in dimensions:
            A_cpu = np.random.rand(N, D).astype(np.float32)
            X_cpu = np.random.rand(D).astype(np.float32)

            A_gpu = cp.asarray(A_cpu)
            X_gpu = cp.asarray(X_cpu)

            # Warm-up GPU
            func_gpu(A_gpu, X_gpu)
            cp.cuda.Device(0).synchronize()

            # Time CPU
            start = time.time()
            func_cpu(A_cpu, X_cpu)
            print(f"CPU {D}")
            cpu_times.append(time.time() - start)

            # Time GPU
            start = time.time()
            func_gpu(A_gpu, X_gpu)
            print(f"GPU {D}")
            cp.cuda.Device(0).synchronize()
            gpu_times.append(time.time() - start)

        return cpu_times, gpu_times

    # Run benchmarks for all distance functions
    distance_tests = [
        ("L2", distance_l2_cpu, distance_l2_gpu),
        ("Cosine", distance_cosine_cpu, distance_cosine_gpu),
        ("Manhattan", distance_manhattan_cpu, distance_manhattan_gpu),
        ("Dot", distance_dot_cpu, distance_dot_gpu),
    ]

    dimensions = [2 ** i for i in range(1, 11)]  # 2, 4, 8, ..., 1024

    results = {"Dimension": dimensions}

    for name, func_cpu, func_gpu in distance_tests:
        cpu_times, gpu_times = benchmark(func_cpu, func_gpu, dimensions)
        results[f"{name}_CPU"] = cpu_times
        results[f"{name}_GPU"] = gpu_times

    # Write to CSV
    output_file = "distance_benchmarks.csv"
    fieldnames = ["Dimension"] + [f"{name}_{mode}" for name, _, _ in distance_tests for mode in ("CPU", "GPU")]

    with open(output_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(dimensions)):
            row = {key: results[key][i] for key in results}
            writer.writerow(row)

    print(f"Benchmark results saved to: {output_file}")