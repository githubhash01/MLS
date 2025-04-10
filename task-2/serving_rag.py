import torch
import numpy as np
import cupy as cp
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import queue
import threading
import time
from typing import List, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from task_1_code import our_kmeans, our_ann, our_knn_cupy

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

app = FastAPI()

# Configuration constants - optimized based on testing
MAX_BATCH_SIZE = 2  # Reduced from 3 to avoid large batch delays
MAX_WAITING_TIME = 0.5  # Reduced from 1.5 to minimize waiting time
REQUEST_TIMEOUT = 20  # Reduced from 30 to prevent long waits

# Create request queue and results store with size limits
request_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
results_store = {}
results_lock = threading.Lock()  # Add lock for thread-safe access to results_store

# Add a thread pool for parallel processing within batches
batch_thread_pool = ThreadPoolExecutor(max_workers=4)  # Increased from 3 to better handle requests

# Example documents in memory
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]


llm_device="cuda:0"
embed_device="cuda:0"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(embed_device)

chat_pipeline = pipeline("text-generation", model="facebook/opt-125m", device=llm_device)

def get_embedding(text: str) -> cp.ndarray:
    """Get embeddings for documents and return a CuPy array (on GPU)"""
    try:
        inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
        # Move inputs to correct device
        inputs = {k: v.to(embed_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = embed_model(**inputs)
        np_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return cp.asarray(np_embedding)
    except Exception as e:
        print(f"Embedding error: {e}")
        # Return a zero embedding as fallback
        return cp.zeros((1, embed_model.config.hidden_size))

# Precompute document embeddings
print("Precomputing document embeddings...")
doc_embeddings = cp.vstack([get_embedding(doc) for doc in documents])
print("Document embeddings completed")

A = doc_embeddings
num_clusters = 1

# precompute kmeans for document set
centroids, labels = our_kmeans(A.shape[0], A.shape[1], A, num_clusters, distance_fn=distance_cosine_gpu, centroid_distance_fn=distance_cosine_kmeans)
print("Processed K-Means")

# Retrieve top K documents from document set (using ANN)
def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    try:
        approx_indices, _ = our_ann(A.shape[0], A.shape[1], A, query_emb, k, centroids, labels, distance_cosine_gpu, distance_cosine_kmeans, num_clusters)
        print(approx_indices)
        print(type(approx_indices))
        print(f"\nGOT DOCUMENTS: {[documents[int(i)] for i in approx_indices]}\n")
        return [documents[int(i)] for i in approx_indices]
    except Exception as e:
        print(f"Retrieval error: {e}")
        # Return first document as fallback with low similarity
        return [(documents[0], 0.2)]

def rag_pipeline(query: str, k: int = 2) -> str:
    query_emb = get_embedding(query)
    
    retrieved_docs = retrieve_top_k(query_emb, k)
    
    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
        
    answer = chat_pipeline(prompt, max_length=100, do_sample=True)[0]["generated_text"]
    print(f"\nGENERATED: {answer}\n")
    return answer

def process_single_request(req: Dict) -> Dict:
    """Process a single request and return the result."""
    try:
        start_time = time.time()
        result = rag_pipeline(req['query'], req['k'])
        processing_time = time.time() - start_time
        print(f"Processed query '{req['query']}' in {processing_time:.2f}s")
        return {
            "status": "complete",
            "result": result,
            "processing_time": processing_time
        }
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return {
            "status": "error",
            "result": f"Error processing your request: {type(e).__name__}"
        }

def process_batch(requests: List[Dict]) -> None:
    """Process a batch of requests in parallel using thread pool."""
    try:
        print(f"Processing batch of {len(requests)} requests")
        start_time = time.time()
        
        # Submit all requests to thread pool
        futures = [batch_thread_pool.submit(process_single_request, req) for req in requests]
        
        # Store results as they complete
        for req, future in zip(requests, futures):
            try:
                result = future.result(timeout=REQUEST_TIMEOUT-2)  # Leave 2s buffer
                with results_lock:
                    results_store[req['id']] = result
            except Exception as e:
                print(f"Future error for {req['id']}: {str(e)}")
                with results_lock:
                    results_store[req['id']] = {
                        "status": "error",
                        "result": f"Error processing your request: {type(e).__name__}"
                    }
        
        batch_time = time.time() - start_time
        print(f"Batch processing completed in {batch_time:.2f}s")
    except Exception as e:
        print(f"Batch processing error: {str(e)}")

def batch_processor():
    """Background thread function to process batched requests."""
    while True:
        batch = []
        try:
            # Try to get the first request with a short timeout
            try:
                first_request = request_queue.get(timeout=0.1)
                batch.append(first_request)
                
                # Try to fill the batch up to MAX_BATCH_SIZE or until MAX_WAITING_TIME
                batch_start_time = time.time()
                while len(batch) < MAX_BATCH_SIZE and time.time() - batch_start_time < MAX_WAITING_TIME:
                    try:
                        request = request_queue.get_nowait()
                        batch.append(request)
                    except queue.Empty:
                        break
                
                # Process the batch (even if only one request)
                if batch:
                    process_batch(batch)
            except queue.Empty:
                # No requests in queue, continue waiting
                continue
                
        except Exception as e:
            print(f"Error in batch processor: {str(e)}")
            # Don't let errors stop the processor
            time.sleep(0.1)  # Short sleep to prevent CPU spinning
            continue
        finally:
            # Clean up old results (anything older than 2x REQUEST_TIMEOUT)
            try:
                current_time = time.time()
                with results_lock:
                    expired_keys = [k for k, v in results_store.items() 
                                    if current_time - float(k.split('_')[0]) > 2*REQUEST_TIMEOUT]
                    for k in expired_keys:
                        results_store.pop(k, None)
            except Exception as e:
                print(f"Error cleaning results: {str(e)}")

# Start the background thread
batch_thread = threading.Thread(target=batch_processor, daemon=True)
batch_thread.start()

# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 2

@app.post("/rag")
async def predict(payload: QueryRequest):
    try:
        # Generate a unique request ID with timestamp
        request_id = f"{time.time()}_{hash(payload.query)}"
        
        # Create the request object
        request = {
            "id": request_id,
            "query": payload.query,
            "k": payload.k
        }
        
        # Try to add request to queue with timeout
        try:
            request_queue.put(request, timeout=1)
            print(f"Added request to queue: {payload.query}")
        except queue.Full:
            return {
                "query": payload.query,
                "result": "Server is overloaded. Please try again later.",
                "status": "error"
            }
        
        # Wait for result with timeout
        start_time = time.time()
        while time.time() - start_time < REQUEST_TIMEOUT:
            with results_lock:
                if request_id in results_store:
                    result = results_store.pop(request_id)  # Remove result after retrieval
                    return {
                        "query": payload.query,
                        "result": result["result"],
                        "status": result["status"]
                    }
            await asyncio.sleep(0.1)
        
        # If we reach here, the request timed out
        return {
            "query": payload.query,
            "result": "Request timed out. Please try again later.",
            "status": "timeout"
        }
    except Exception as e:
        print(f"API endpoint error: {str(e)}")
        return {
            "query": payload.query,
            "result": f"Error processing your request: {type(e).__name__}",
            "status": "error"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
