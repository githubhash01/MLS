import numpy as np
import cupy as cp

import time
import csv

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

def benchmark(func_cpu, func_gpu, dimensions, N=4096):
    cpu_times = []
    gpu_times = []

    for D in dimensions:
        A_cpu = np.random.rand(N, D)#.astype(np.float32)
        X_cpu = np.random.rand(D)#.astype(np.float32)

        A_gpu = cp.asarray(A_cpu)
        X_gpu = cp.asarray(X_cpu)

        # Time CPU
        start = time.time()
        func_cpu(A_cpu, X_cpu)
        print(f"CPU {D}")
        cpu_times.append(time.time() - start)



        # Warm-up GPU
        func_gpu(A_gpu, X_gpu)
        cp.cuda.Device(0).synchronize()

        
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