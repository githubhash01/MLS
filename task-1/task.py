# import torch
import cupy as cp
import cupyx
# import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

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

def our_knn_cupy(N, D, A, X, K):
    """
    Input:
        N: Number of vectors
        D: Dimension of vectors
        A[N, D]: A collection of vectors
        X: A specified vector
        K: Top K
    """
    X = X.reshape(1, D)
    # with cupyx.profiler.time_range("KNN Euclidean Distances", color_id=0):
    distances = cp.linalg.norm(A - X, axis=1)
    # with cupyx.profiler.time_range("KNN Argsorting", color_id=0):
    top_k_indices = cp.argpartition(distances, K)[:K]
    sorted_k_indices = top_k_indices[cp.argsort(distances[top_k_indices])]
        # indices = cp.argsort(distances)[:K]
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


def our_knn_cpu(N, D, A, X, K):
    """
    Input:
        N: Number of vectors
        D: Dimension of vectors
        A[N, D]: A collection of vectors (NumPy array)
        X: A specified vector (NumPy array of shape [D])
        K: Top K (number of nearest neighbors to find)
    
    Output:
        indices: The indices of the K-nearest neighbors in the array A
        distances: The corresponding distances of the K-nearest neighbors
    """
    # Ensure X has the correct shape (1, D) for broadcasting
    X = X.reshape(1, D)
    
    # Step 1: Calculate the squared Euclidean distances between X and all vectors in A
    # Using broadcasting: ||A - X||^2 = sum((A - X)^2) along axis=1
    distances = np.linalg.norm(A - X, axis=1)
    
    # Step 2: Get the indices of the K smallest distances
    indices = np.argsort(distances)[:K]
    
    # Step 3: Gather the K smallest distances
    k_distances = distances[indices]

    return indices, k_distances


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

def measure_speedup_knn(N, D, K):
    # Generate random data
    A_gpu = cp.random.rand(N, D)  # CuPy array for GPU
    X_gpu = cp.random.rand(D)     # CuPy vector for GPU

    A_cpu = np.random.rand(N, D)  # NumPy array for CPU
    X_cpu = np.random.rand(D)     # NumPy vector for CPU

    our_knn_raw(N, D, A_gpu, X_gpu, K)

    # Measure time for the GPU implementation
    start_gpu = time.perf_counter()
    indices_gpu, distances_gpu = our_knn_raw(N, D, A_gpu, X_gpu, K)
    end_gpu = time.perf_counter()
    gpu_time = end_gpu - start_gpu

    our_knn_raw_tiled(N, D, A_gpu, X_gpu, K)

    # Measure time for the CPU implementation
    start_cpu = time.perf_counter()
    indices_cpu, distances_cpu = our_knn_raw_tiled(N, D, A_gpu, X_gpu, K)
    end_cpu = time.perf_counter()
    cpu_time = end_cpu - start_cpu

    # Calculate and print the speedup
    speedup = cpu_time / gpu_time
    print(f"CPU Time = {cpu_time:.6f}s, GPU Time = {gpu_time:.6f}s, Speedup = {speedup:.2f}x")
    assert cp.allclose(indices_gpu, indices_gpu, atol=1e-6), "Mismatch in results!"


def benchmark_knn(func, N, D, A, X, K, runs=5):
    # Warm-up run (to avoid startup overhead

    func(N, D, A, X, K)

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
    time_raw = benchmark_knn(our_knn_raw_tiled, N, D, A, X, K)

    print(f"Pure CuPy Time: {time_cupy:.4f} ms")
    print(f"Raw Kernel Time: {time_raw:.4f} ms")
    print(f"Speedup: {time_cupy / time_raw:.2f}x")

def profile_gpu_knn(func, N, D, K):
    cp.random.seed(12345)

    A = cp.random.rand(N, D)  # CuPy array for GPU
    X = cp.random.rand(D)     # CuPy vector for GPU

    indices, _ = func(N, D, A, X, K)
    print(indices)

    return

# Run the speedup measurements
if __name__ == "__main__":
    # test_distances()
    # measure_speedup_knn(2**9, 10, 5)
    # profile_gpu_knn(our_knn_cupy, 20, 2**20, 10)
    profile_gpu_knn(our_knn_raw_tiled, 2**15, 20, 10)

    # profile_knn(our_knn_raw, 100000, 10, 5)
    # new_benchmark_knn(2**15, 20, 10)
