import requests
import time
import concurrent.futures
from typing import Dict, Any
import json

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
]

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

def test_batch_processing():
    """Test batch processing with concurrent requests."""
    print("Starting batch processing test...")
    print(f"Sending {len(TEST_QUERIES)} concurrent requests...")
    
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
                print(f"\nQuery: {query}")
                print(f"Response time: {result['time']:.2f} seconds")
                print(f"Status: {result['status']}")
                if isinstance(result['response'], dict):
                    print(f"Result: {result['response'].get('result', 'No result')}")
            except Exception as e:
                print(f"Query {query} generated an exception: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average response time: {sum(r['time'] for r in results)/len(results):.2f} seconds")
    print(f"Successful requests: {sum(1 for r in results if r['status'] == 200)}/{len(results)}")
    
    # Save results to file
    with open("batch_test_results.json", "w") as f:
        json.dump({
            "total_time": total_time,
            "results": results
        }, f, indent=2)

if __name__ == "__main__":
    test_batch_processing() 