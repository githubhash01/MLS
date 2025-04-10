import requests
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

def send_request(query, k=2, host='localhost', port=8000):
    """Send a query to the RAG service and return the result"""
    url = f"http://{host}:{port}/rag"
    payload = {"query": query, "k": k}
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=30)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"Query: {query}")
            print(f"Result: {result['result']}")
            print(f"Response time: {elapsed:.2f} seconds")
            print(f"Status: {result['status']}")
            print("-" * 40)
            return True, result, elapsed
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False, None, elapsed
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False, None, 0

def run_single_test(host='localhost', port=8000):
    """Run a single test query"""
    print("\n=== Running single test query ===")
    query = "What animals can hover in the air?"
    success, result, elapsed = send_request(query, host=host, port=port)
    return success

def run_parallel_tests(n_requests=5, host='localhost', port=8000):
    """Run multiple requests in parallel"""
    print(f"\n=== Running {n_requests} parallel requests ===")
    
    test_queries = [
        "Tell me about cats",
        "What animals can hover in the air?",
        "Tell me about dogs",
        "What do cats look like?",
        "Are dogs wild animals?",
        "How do hummingbirds fly?",
        "What are mammals?",
        "Do cats make good pets?",
        "Tell me about birds",
        "What animals are domesticated?"
    ]
    
    # Use only as many queries as requested, cycling through if needed
    queries = [test_queries[i % len(test_queries)] for i in range(n_requests)]
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=min(10, n_requests)) as executor:
        results = list(executor.map(
            lambda q: send_request(q, host=host, port=port), 
            queries
        ))
    
    total_time = time.time() - start_time
    successes = sum(1 for success, _, _ in results if success)
    
    print(f"\nResults: {successes}/{n_requests} successful requests")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average response time: {total_time/n_requests:.2f} seconds")
    
    return successes == n_requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the RAG service")
    parser.add_argument("--host", default="localhost", help="Service host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Service port (default: 8000)")
    parser.add_argument("--parallel", type=int, default=5, help="Number of parallel requests (default: 5)")
    
    args = parser.parse_args()
    
    print(f"Testing RAG service at http://{args.host}:{args.port}")
    
    # First run a single request to check if the service is up
    if run_single_test(host=args.host, port=args.port):
        # Then run parallel tests
        run_parallel_tests(args.parallel, host=args.host, port=args.port)
    else:
        print("Single test failed. Please check if the service is running properly.") 