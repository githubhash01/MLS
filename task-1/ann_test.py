import numpy as np
import cupy as cp
import time
import pandas as pd

from task_ import our_ann, our_ann_cpu, distance_l2_cpu, distance_l2_kmeans_cpu, distance_l2_gpu, distance_l2_kmeans_gpu

import numpy as np
import cupy as cp
import time
import pandas as pd

def benchmark_ann_runtimes_to_csv(
    N_values, D_values, K, num_clusters,
    ann_cpu_fn, ann_gpu_fn,
    distance_fn_cpu, distance_fn_gpu,
    centroid_distance_fn_cpu, centroid_distance_fn_gpu,
    output_csv_path="ann_runtime_results.csv"
):
    results = []

    for N in N_values:
        for D in D_values:
            print(f"Testing N={N}, D={D}...")

            # Generate synthetic data
            np.random.seed(0)
            A_cpu = np.random.randn(N, D).astype(np.float32)
            X_cpu = np.random.randn(D).astype(np.float32)

            centroid_indices = np.random.choice(N, num_clusters, replace=False)
            centroids_cpu = A_cpu[centroid_indices]
            labels_cpu = np.random.randint(0, num_clusters, size=N)

            # GPU versions
            A_gpu = cp.asarray(A_cpu)
            X_gpu = cp.asarray(X_cpu)
            centroids_gpu = cp.asarray(centroids_cpu)
            labels_gpu = cp.asarray(labels_cpu)

            # --- Time CPU
            start_cpu = time.time()
            ann_cpu_fn(N, D, A_cpu, X_cpu, K, centroids_cpu, labels_cpu,
                       distance_fn=distance_fn_cpu,
                       centroid_distance_fn=centroid_distance_fn_cpu,
                       num_clusters=num_clusters)
            cpu_time = time.time() - start_cpu

            # --- Time GPU
            start_gpu = time.time()
            ann_gpu_fn(N, D, A_gpu, X_gpu, K, centroids_gpu, labels_gpu,
                       distance_fn=distance_fn_gpu,
                       centroid_distance_fn=centroid_distance_fn_gpu,
                       num_clusters=num_clusters)
            cp.cuda.Device().synchronize()
            gpu_time = time.time() - start_gpu

            # Save result
            results.append({
                "N": N,
                "D": D,
                "CPU_Time": cpu_time,
                "GPU_Time": gpu_time
            })

    # Save all results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"\nSaved results to: {output_csv_path}")


N_values = [2**i for i in range(12, 20, 2)]
# N_values = [2**12, 2**14, 2**16]
D_values = [2**i for i in range(6, 9)]
K = 5
num_clusters = 20

benchmark_ann_runtimes_to_csv(
    N_values,
    D_values,
    K,
    num_clusters,
    ann_cpu_fn=our_ann_cpu,
    ann_gpu_fn=our_ann,
    distance_fn_cpu=distance_l2_cpu,
    distance_fn_gpu=distance_l2_gpu,
    centroid_distance_fn_cpu=distance_l2_kmeans_cpu,
    centroid_distance_fn_gpu=distance_l2_gpu,
    output_csv_path="ann_runtime_results.csv"
)