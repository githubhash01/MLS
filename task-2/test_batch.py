import requests
import time
import threading
import random
import concurrent.futures
from typing import Dict, Any, List
import json
import numpy as np

# Server configuration
BASE_URL = "http://localhost:8000/rag"
HEADERS = {"Content-Type": "application/json"}

# Test queries
TEST_QUERIES = [
    "Which animals can hover in the air?",
    "Tell me about cats",
    "What do dogs do?",
    "Can birds fly?",
    "How do hummingbirds move?",
    "What are domestic pets?",
    "Tell me about mammals",
    "How do animals move?",

    "Which animals can hover in the air?",
    "Tell me about cats",
    "What do dogs do?",
    "Can birds fly?",
    "How do hummingbirds move?",
    "What are domestic pets?",
    "Tell me about mammals",
    "How do animals move?",

    "Which animals can hover in the air?",
    "Tell me about cats",
    "What do dogs do?",
    "Can birds fly?",
    "How do hummingbirds move?",
    "What are domestic pets?",
    "Tell me about mammals",
    "How do animals move?",

    "Which animals can hover in the air?",
    "Tell me about cats",
    "What do dogs do?",
    "Can birds fly?",
    "How do hummingbirds move?",
    "What are domestic pets?",
    "Tell me about mammals",
    "How do animals move?"
] * 1

def send_request(query: str) -> Dict[str, Any]:
    """Send a single request to the RAG service."""
    payload = {"query": query}
    try:
        start_time = time.time()
        response = requests.post(BASE_URL, headers=HEADERS, json=payload)
        end_time = time.time()
        
        return {
            "query": query,
            "status": response.status_code,
            "response": response.json(),
            "time": end_time - start_time
        }
    except Exception as e:
        return {
            "query": query,
            "status": "error",
            "response": str(e),
            "time": -1
        }
    

def run_requests(queries: List[str], rate: float, mode: str = "uniform"):
    """
    Send requests at the specified rate using uniform or Poisson distribution.

    Args:
        queries: List of query strings
        rate: Requests per second
        mode: "uniform" or "poisson"
    """
    interval_fn = (
        lambda: 1.0 / rate if mode == "uniform" else random.expovariate(rate)
    )

    results = []

    def request_thread(query):
        result = send_request(query)
        results.append({"latency":result["time"]})
        # print(f"[{result['status']}] {result['query']} in {result['time']:.2f}s")
        # results.append(result)

    start = time.time()
    for query in queries:
        threading.Thread(target=request_thread, args=(query,)).start()
        # time.sleep(interval_fn())
    end = time.time()
    return results, start, end

# # ========== MAIN ==========
# if __name__ == "__main__":
#     # parser = argparse.ArgumentParser(description="Send RAG requests at controlled rate.")
#     # parser.add_argument("--rate", type=float, required=True, help="Requests per second")
#     # parser.add_argument("--mode", type=str, choices=["uniform", "poisson"], default="uniform")
#     # parser.add_argument("--count", type=int, default=10, help="Number of requests to send")
    
#     # args = parser.parse_args()

#     # sample_queries = [f"What is the answer to question {i}?" for i in range(args.count)]
    
#     # print(f"Sending {args.count} requests at {args.rate} rps using {args.mode} intervals.")
#     res, start, end = run_requests(TEST_QUERIES, 30, "uniform")

#     print(f"Throughput: {len(res) / (end-start)}")
#     print(f"Avg. Latency: {np.mean([r['latency'] for r in res])}")





def test_batch_processing():
    """Test batch processing with concurrent requests."""
    # print("Starting batch processing test...")
    # print(f"Sending {len(TEST_QUERIES)} concurrent requests...")
    
    results = []
    start_time = time.time()
    
    # Use ThreadPoolExecutor to send concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(TEST_QUERIES)) as executor:
        future_to_query = {executor.submit(send_request, query): query for query in TEST_QUERIES}
        
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                result = future.result()
                results.append(result)
                # print(f"\nQuery: {query}")
                print(f"Response time: {result['time']:.2f} seconds")
                # print(f"Status: {result['status']}")
                # if isinstance(result['response'], dict):
                    # print(f"Result: {result['response'].get('result', 'No result')}")
            except Exception as e:
                print(f"Query {query} generated an exception: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average response time: {sum(r['time'] for r in results)/len(results):.2f} seconds")
    print(f"Successful requests: {sum(1 for r in results if r['status'] == 200)}/{len(results)}")
    print(f"Throughput: {len(results) / total_time} req/s")
    # Save results to file
    with open("batch_test_results.json", "w") as f:
        json.dump({
            "total_time": total_time,
            "results": results
        }, f, indent=2)

if __name__ == "__main__":
    test_batch_processing() 

