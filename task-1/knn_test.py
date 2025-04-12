import numpy as np
import cupy as cp

import time
import csv

from task_ import our_knn_cpu, our_knn_gpu

from distance_functions import (
    distance_cosine_cpu, 
    distance_cosine_gpu, 
    distance_dot_cpu, 
    distance_dot_gpu, 
    distance_l2_cpu, 
    distance_l2_gpu, 
    distance_manhattan_cpu, 
    distance_manhattan_gpu
)

# Setup
D = 128
K = 10
sizes = [2 ** i for i in range(12, 23)]  # 4096 to 4,194,304

distance_funcs = {
    "L2": (distance_l2_cpu, distance_l2_gpu),
    "Cosine": (distance_cosine_cpu, distance_cosine_gpu),
    "Manhattan": (distance_manhattan_cpu, distance_manhattan_gpu),
    "Dot": (distance_dot_cpu, distance_dot_gpu),
}

# Results dictionary
results = {"N": sizes}

for name in distance_funcs:
    results[f"{name}_CPU"] = []
    results[f"{name}_GPU"] = []

def benchmark_all_knn():
    for N in sizes:
        print(f"\nDataset size: N = {N}")
        A_cpu = np.random.rand(N, D).astype(np.float32)
        X_cpu = np.random.rand(D).astype(np.float32)
        A_gpu = cp.asarray(A_cpu)
        X_gpu = cp.asarray(X_cpu)

        for name, (cpu_func, gpu_func) in distance_funcs.items():
            print(f"  Testing {name}...")

            # Warm-up GPU
            our_knn_gpu(N, D, A_gpu, X_gpu, K, distance_func=gpu_func)
            cp.cuda.Device(0).synchronize()

            # CPU time
            start = time.time()
            our_knn_cpu(N, D, A_cpu, X_cpu, K, distance_func=cpu_func)
            cpu_time = time.time() - start

            # GPU time
            start = time.time()
            our_knn_gpu(N, D, A_gpu, X_gpu, K, distance_func=gpu_func)
            cp.cuda.Device(0).synchronize()
            gpu_time = time.time() - start

            results[f"{name}_CPU"].append(cpu_time)
            results[f"{name}_GPU"].append(gpu_time)

            print(f"    CPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s")

# Run benchmarks
benchmark_all_knn()

# --------------------------------------
# Save results to CSV
# --------------------------------------

output_file = "knn_all_distances_benchmarks.csv"
fieldnames = list(results.keys())

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(sizes)):
        row = {key: results[key][i] for key in results}
        writer.writerow(row)

print(f"\n✅ All benchmarks saved to: {output_file}")