import cupy as cp
import numpy as np


# GPU Functions (CuPy)
def distance_l2_gpu(X, Y):
    return cp.linalg.norm(X - Y, axis=1)

# def distance_cosine_gpu(X, Y):
#     X_norm = cp.linalg.norm(X, axis=1)
#     Y_norm = cp.linalg.norm(Y)
#     dot = X @ Y.T
#     return 1 - (dot / (X_norm * Y_norm + 1e-8))  # Avoid divide by zero

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


# CPU Functions (NumPy)

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






# KMeans

def distance_l2_kmeans(A, C):
    # A: [N, D], C: [K, D]
    return cp.linalg.norm(A[:, cp.newaxis, :] - C[cp.newaxis, :, :], axis=2)

# def distance_cosine_kmeans(A, C):
#     A_norm = cp.linalg.norm(A, axis=1, keepdims=True)  # Shape [N, 1]
#     C_norm = cp.linalg.norm(C, axis=1, keepdims=True)  # Shape [K, 1]
    
#     sim = cp.dot(A, C.T) / (A_norm * C_norm.T + 1e-8)  # Shape [N, K]
    
#     return 1 - sim  # Cosine distance, shape [N, K]

def distance_cosine_kmeans(A, C):
    A_norm = cp.linalg.norm(A, axis=1, keepdims=True)  # Shape [N, 1]
    C_norm = cp.linalg.norm(C, axis=1, keepdims=True)  # Shape [K, 1]
    
    sim = cp.dot(A, C.T) / (A_norm * C_norm.T + 1e-8)  # Shape [N, K]
    
    return 1 - sim  # Cosine distance, shape [N, K]

def distance_dot_kmeans(A, C):
    """Dot similarity (negated for distance)"""
    return -A @ C.T  # Higher dot product = closer

def distance_manhattan_kmeans(A, C):
    """Manhattan (L1) distance between A and centroids"""
    return cp.sum(cp.abs(A[:, cp.newaxis, :] - C[cp.newaxis, :, :]), axis=2)
