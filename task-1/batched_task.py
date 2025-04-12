import cupy as cp
from distance_functions import distance_cosine_kmeans

def distance_kmeans_cupy_l2(A_batch, centers):
    return cp.linalg.norm(A_batch[:, cp.newaxis, :] - centers[cp.newaxis, :, :], axis=2)

def update_centers_cupy(N, D, A_gpu, K, assignments):
    sums = cp.zeros((K, D), dtype=A_gpu.dtype)
    cp.add.at(sums, assignments, A_gpu)
    counts = cp.bincount(assignments, minlength=K)
    return sums / counts[:, None]

def maximin_cupy_batched(N, D, A_gpu, K, dist_func, batch_size):
    # random point for the first center.
    first_idx = int(cp.random.choice(N, 1).item())
    centers = A_gpu[first_idx:first_idx + 1].copy()  # shape (1, D)

    for i in range(1, K):
        min_dists = cp.empty(N, dtype=cp.float32)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            dists_batch = dist_func(A_gpu[start:end], centers)

            # For each point in the batch, take the minimum distance among the centers.
            min_dists[start:end] = cp.min(dists_batch, axis=1)

        # Choose the point with the maximum minimum distance and append point to centers
        next_idx = int(cp.argmax(min_dists).item())
        centers = cp.concatenate([centers, A_gpu[next_idx:next_idx + 1]], axis=0)

    return centers

def kmeans_cupy_batched(N, D, A, K, max_iter=100, batch_size=1_000_000, dist_func=distance_cosine_kmeans):
    # Upload full dataset onto GPU (hopefully this is ok)
    A_gpu = cp.asarray(A, dtype=cp.float32)

    # Get initial centers using a batched version of maximin
    centers = maximin_cupy_batched(N, D, A_gpu, K, dist_func, batch_size)
    assignments = cp.full(N, -1, dtype=cp.int32)

    for it in range(max_iter):
        new_assignments = cp.empty(N, dtype=cp.int32)

        # Compute distances in batches
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            # build up the distances for the current batch
            distances_batch = dist_func(A_gpu[start:end], centers)
            new_assignments[start:end] = cp.argmin(distances_batch, axis=1)

            # free temporary arrays
            cp.get_default_memory_pool().free_all_blocks()

        # Check for convergence
        if cp.array_equal(assignments, new_assignments):
            break

        assignments = new_assignments
        centers = update_centers_cupy(N, D, A_gpu, K, assignments)

    return cp.asnumpy(centers)