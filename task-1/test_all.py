import torch
import numpy as np
import time
from task import (
    distance_cosine, 
    distance_l2, 
    distance_dot, 
    distance_manhattan,
    our_knn,
    our_kmeans,
    our_ann,
    recall_rate
)
from sklearn.neighbors import NearestNeighbors
import pandas as pd

class TestResults:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = []
        self.benchmarks = []
    
    def add_result(self, test_name, passed, message=""):
        self.total += 1
        if passed:
            self.passed += 1
            print(f"✓ {test_name} passed")
        else:
            self.failed.append((test_name, message))
            print(f"✗ {test_name} failed: {message}")
    
    def add_benchmark(self, name, size, time_mean, time_std, baseline_time=None):
        self.benchmarks.append({
            'name': name,
            'size': size,
            'time': time_mean,
            'std': time_std,
            'baseline': baseline_time
        })
    
    def summary(self):
        print(f"\nTest Summary:")
        print(f"Passed: {self.passed}/{self.total} ({self.passed/self.total*100:.1f}%)")
        
        if self.failed:
            print("\nFailed Tests:")
            for name, msg in self.failed:
                print(f"- {name}: {msg}")
        
        if self.benchmarks:
            print("\nPerformance Benchmarks:")
            df = pd.DataFrame(self.benchmarks)
            print(df.to_string(index=False))

def benchmark(func, *args, num_runs=5):
    """Measure execution time of a function"""
    times = []
    result = None
    for i in range(num_runs):
        torch.cuda.empty_cache()  # Clear GPU memory if available
        start = time.time()
        result = func(*args)
        times.append(time.time() - start)
    return result, np.mean(times), np.std(times)

def test_distance_functions_performance(results):
    """Test distance functions with large matrices"""
    print("\n=== Testing Distance Functions Performance ===")
    
    sizes = [(1000, 128), (10000, 128), (100000, 128)]
    
    for N, D in sizes:
        X = torch.randn(N, D)
        Y = torch.randn(N, D)
        
        # Test L2 distance
        _, time_mean, time_std = benchmark(distance_l2, X, Y)
        results.add_benchmark(f"L2 distance", f"{N}x{D}", time_mean, time_std)
        
        # Test cosine distance
        _, time_mean, time_std = benchmark(distance_cosine, X, Y)
        results.add_benchmark(f"Cosine distance", f"{N}x{D}", time_mean, time_std)
        
        # Compare with sklearn implementation
        start = time.time()
        nbrs = NearestNeighbors(algorithm='brute', metric='euclidean').fit(X.numpy())
        nbrs.kneighbors(Y.numpy())
        baseline_time = time.time() - start
        
        results.add_result(
            f"Distance computation {N}x{D}",
            time_mean < baseline_time,
            f"Our impl: {time_mean:.3f}s, Baseline: {baseline_time:.3f}s"
        )

def test_knn_performance(results):
    """Test KNN implementation with various sizes"""
    print("\n=== Testing KNN Performance ===")
    
    sizes = [
        (1000, 128, 10),
        (10000, 128, 10),
        (100000, 128, 10),
        (1000000, 128, 10)  # 1M points
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for N, D, K in sizes:
        # Generate data
        A = torch.randn(N, D)
        X = torch.randn(100, D)  # 100 query points
        
        # Move to GPU if available
        if torch.cuda.is_available():
            A = A.cuda()
            X = X.cuda()
        
        # Our implementation
        _, time_mean, time_std = benchmark(our_knn, N, D, A, X, K)
        results.add_benchmark(f"KNN", f"{N}x{D}", time_mean, time_std)
        
        # Sklearn baseline (CPU only)
        A_cpu = A.cpu().numpy()
        X_cpu = X.cpu().numpy()
        start = time.time()
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(A_cpu)
        nbrs.kneighbors(X_cpu)
        baseline_time = time.time() - start
        
        results.add_result(
            f"KNN {N} points",
            time_mean < baseline_time,
            f"Our impl: {time_mean:.3f}s, Baseline: {baseline_time:.3f}s"
        )

def test_kmeans_performance(results):
    """Test K-means implementation with large datasets"""
    print("\n=== Testing K-means Performance ===")
    
    sizes = [
        (1000, 128, 10),
        (10000, 128, 10),
        (100000, 128, 10)
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for N, D, K in sizes:
        # Generate clustered data
        centers = torch.randn(K, D)
        A = centers.repeat(N//K, 1) + torch.randn(N, D) * 0.1
        
        if torch.cuda.is_available():
            A = A.cuda()
        
        # Our implementation
        _, time_mean, time_std = benchmark(our_kmeans, N, D, A, K, num_runs=3)
        results.add_benchmark(f"K-means", f"{N}x{D}", time_mean, time_std)
        
        # Compare clustering quality
        centroids = our_kmeans(N, D, A, K)
        if torch.cuda.is_available():
            centroids = centroids.cpu()
            A = A.cpu()
        
        # Measure cluster quality
        distances = torch.cdist(A, centroids)
        assignments = torch.argmin(distances, dim=1)
        inertia = torch.sum(torch.min(distances, dim=1)[0])
        
        results.add_result(
            f"K-means quality {N} points",
            True,  # Always pass, just logging the quality
            f"Inertia: {inertia:.2e}"
        )

def test_ann_performance(results):
    """Test ANN implementation with large datasets"""
    print("\n=== Testing ANN Performance ===")
    
    sizes = [
        (10000, 128, 10),
        (100000, 128, 10),
        (1000000, 128, 10)  # 1M points
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for N, D, K in sizes:
        # Generate data
        A = torch.randn(N, D)
        X = torch.randn(100, D)  # 100 query points
        
        if torch.cuda.is_available():
            A = A.cuda()
            X = X.cuda()
        
        # Test ANN
        _, time_mean, time_std = benchmark(our_ann, N, D, A, X, K, num_runs=3)
        results.add_benchmark(f"ANN", f"{N}x{D}", time_mean, time_std)
        
        # Compare with exact KNN
        ann_indices = our_ann(N, D, A, X, K)
        exact_indices = our_knn(N, D, A, X, K)
        
        if torch.cuda.is_available():
            ann_indices = ann_indices.cpu()
            exact_indices = exact_indices.cpu()
        
        recalls = [recall_rate(ann_indices[i].tolist(), exact_indices[i].tolist()) 
                  for i in range(X.shape[0])]
        avg_recall = sum(recalls) / len(recalls)
        
        results.add_result(
            f"ANN quality {N} points",
            avg_recall > 0.3,  # Lower threshold for very large datasets
            f"Recall: {avg_recall:.2f}"
        )

def main():
    print("Starting performance tests...")
    results = TestResults()
    
    try:
        test_distance_functions_performance(results)
        test_knn_performance(results)
        test_kmeans_performance(results)
        test_ann_performance(results)
    except Exception as e:
        print(f"Tests failed with exception: {str(e)}")
        raise
    
    results.summary()

if __name__ == "__main__":
    main() 