import numpy as np
import json
import os
import torch
from task import test_kmeans, test_knn, test_ann, our_kmeans, our_knn, our_ann
from test import testdata_kmeans, testdata_knn, testdata_ann
import time
import matplotlib.pyplot as plt


def generate_test_files():
    # Set a seed for reproducibility
    np.random.seed(0)
    
    # Define parameters for the synthetic dataset
    N = 1000    # number of data points
    D = 100     # dimensionality
    K = 10      # number of clusters/nearest neighbors
    num_queries = 3  # number of query points for knn/ann tests
    
    # Generate clustered dataset A with K clusters in high-dimensional space
    centers = np.random.uniform(low=-10, high=10, size=(K, D))
    points_per_cluster = N // K
    remainder = N % K
    A_list = []
    for i in range(K):
         n_points = points_per_cluster + (1 if i < remainder else 0)
         points = centers[i] + 0.5 * np.random.randn(n_points, D)
         A_list.append(points)
    A = np.vstack(A_list)
    a_file = "a_file.txt"
    np.savetxt(a_file, A)
    
    # Generate query data X from cluster centers: choose random clusters and add small noise
    chosen_clusters = np.random.choice(K, num_queries)
    X = centers[chosen_clusters] + 0.5 * np.random.randn(num_queries, D)
    x_file = "x_file.txt"
    np.savetxt(x_file, X)
    
    # Create a JSON configuration file with required fields for the tests
    test_json = {
        "n": N,
        "d": D,
        "a_file": a_file,
        "x_file": x_file,
        "k": K
    }
    test_file = "test_file.json"
    with open(test_file, "w") as f:
        json.dump(test_json, f, indent=4)
    
    print(f"Generated test file: {test_file}")
    print(f"Generated data file: {a_file}")
    print(f"Generated query file: {x_file}")


def run_tests():
    print("Running k-means test:")
    test_kmeans()
    
    print("\nRunning k-NN test:")
    test_knn()
    
    print("\nRunning ANN test:")
    test_ann()


def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)


def benchmark_kmeans(runs=10):
    N, D, A, K = testdata_kmeans("test_file.json")
    times = []
    for i in range(runs):
        start = time.time()
        our_kmeans(N, D, A, K)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    avg_time = sum(times) / runs
    print(f"Average k-means time: {avg_time:.6f} sec over {runs} runs")


def benchmark_knn(runs=10):
    N, D, A, X, K = testdata_knn("test_file.json")
    times = []
    for i in range(runs):
        start = time.time()
        our_knn(N, D, A, X, K)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    avg_time = sum(times) / runs
    print(f"Average k-NN time: {avg_time:.6f} sec over {runs} runs")


def benchmark_ann(runs=10):
    N, D, A, X, K = testdata_ann("test_file.json")
    times = []
    for i in range(runs):
        start = time.time()
        our_ann(N, D, A, X, K)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    avg_time = sum(times) / runs
    print(f"Average ANN time: {avg_time:.6f} sec over {runs} runs")


def evaluate_ann():
    N, D, A, X, K = testdata_ann("test_file.json")
    knn_result = our_knn(N, D, A, X, K)
    ann_result = our_ann(N, D, A, X, K)
    recalls = []
    for i in range(knn_result.size(0)):
        knn_list = knn_result[i].tolist()
        ann_list = ann_result[i].tolist()
        rate = len(set(knn_list) & set(ann_list)) / K
        recalls.append(rate)
    avg_recall = sum(recalls) / len(recalls)
    print(f"Average ANN recall: {avg_recall*100:.2f}%")
    for i, r in enumerate(recalls):
        print(f"Recall for query {i}: {r*100:.2f}%")


def plot_benchmark_averages(runs=10):
    # Benchmark k-means
    N, D, A, K = testdata_kmeans("test_file.json")
    times_kmeans = []
    for i in range(runs):
        start = time.time()
        our_kmeans(N, D, A, K)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        times_kmeans.append(end - start)
    avg_kmeans = sum(times_kmeans) / runs

    # Benchmark k-NN
    N, D, A, X, K = testdata_knn("test_file.json")
    times_knn = []
    for i in range(runs):
        start = time.time()
        our_knn(N, D, A, X, K)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        times_knn.append(end - start)
    avg_knn = sum(times_knn) / runs

    # Benchmark ANN
    N, D, A, X, K = testdata_ann("test_file.json")
    times_ann = []
    for i in range(runs):
        start = time.time()
        our_ann(N, D, A, X, K)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        times_ann.append(end - start)
    avg_ann = sum(times_ann) / runs

    algorithms = ['k-means', 'k-NN', 'ANN']
    avg_times = [avg_kmeans, avg_knn, avg_ann]

    plt.figure(figsize=(8,6))
    x = list(range(len(algorithms)))
    plt.scatter(x, avg_times, color=['blue','green','orange'], s=100)
    plt.xticks(x, algorithms)
    plt.xlabel("Algorithm")
    plt.ylabel("Average Execution Time (sec)")
    plt.title("Benchmark Average Execution Time")
    plt.show()


def plot_ann_recall_bar():
    N, D, A, X, K = testdata_ann("test_file.json")
    knn_result = our_knn(N, D, A, X, K)
    ann_result = our_ann(N, D, A, X, K)
    recalls = []
    for i in range(knn_result.size(0)):
        knn_list = knn_result[i].tolist()
        ann_list = ann_result[i].tolist()
        rate = len(set(knn_list) & set(ann_list)) / K
        recalls.append(rate * 100)  # percentage
    queries = list(range(knn_result.size(0)))
    plt.figure(figsize=(8,6))
    plt.scatter(queries, recalls, color='purple', s=100)
    plt.xlabel("Query Index")
    plt.ylabel("Recall (%)")
    plt.title("ANN Recall per Query")
    plt.show()


def plot_kmeans_clusters():
    """Plots the clusters computed by k-means, projected to 2D using PCA."""
    from task import our_kmeans
    import numpy as np
    import torch

    # Load test data for k-means
    N, D, A, K = testdata_kmeans("test_file.json")

    # Run k-means clustering
    centroids = our_kmeans(N, D, A, K)

    # Convert tensors to numpy arrays
    A_np = A.cpu().numpy() if torch.is_tensor(A) else np.array(A)
    centroids_np = centroids.cpu().numpy() if torch.is_tensor(centroids) else np.array(centroids)

    # Perform PCA on A_np to project to 2D
    mean_X = np.mean(A_np, axis=0)
    X_centered = A_np - mean_X
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:2]  # top 2 components (2 x D)
    proj = np.dot(X_centered, components.T)  # shape (N, 2)

    # Project centroids using the same transformation
    proj_centroids = np.dot(centroids_np - mean_X, components.T)

    # Compute cluster assignments in original space
    distances = np.linalg.norm(A_np[:, None] - centroids_np[None, :], axis=2)
    assignments = np.argmin(distances, axis=1)

    # Plot the clusters and centroids
    plt.figure(figsize=(8,6))
    plt.scatter(proj[:,0], proj[:,1], c=assignments, cmap='viridis', s=20, alpha=0.6)
    plt.scatter(proj_centroids[:,0], proj_centroids[:,1], color='red', marker='X', s=200, label='Centroids')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("K-means Clusters Projection (PCA)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    generate_test_files()

    print("Running k-means test:")
    test_kmeans()
    
    print("\nRunning k-NN test:")
    test_knn()
    
    print("\nRunning ANN test:")
    test_ann()

    print("\nBenchmarking performance:")
    benchmark_kmeans()
    benchmark_knn()
    benchmark_ann()

    print("\nEvaluating ANN recall:")
    evaluate_ann()

    print("\nPlotting Benchmark Averages:")
    plot_benchmark_averages()

    print("\nPlotting ANN Recall per Query:")
    plot_ann_recall_bar()

    print("\nPlotting K-means Clusters Projection (PCA):")
    plot_kmeans_clusters() 