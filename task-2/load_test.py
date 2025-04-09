import requests
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import argparse

class LoadTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        self.error_count = 0
        self.success_count = 0
        self.lock = threading.Lock()

    def send_request(self, query: str) -> Dict[str, Any]:
        """Send a single request and record metrics."""
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/rag",
                headers={"Content-Type": "application/json"},
                json={"query": query},
                timeout=30
            )
            end_time = time.time()
            latency = end_time - start_time
            
            with self.lock:
                if response.status_code == 200:
                    self.success_count += 1
                else:
                    self.error_count += 1
                
                self.results.append({
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "latency": latency,
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                })
            
            return {
                "success": response.status_code == 200,
                "latency": latency,
                "response": response.json() if response.status_code == 200 else None
            }
            
        except Exception as e:
            end_time = time.time()
            with self.lock:
                self.error_count += 1
                self.results.append({
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "latency": end_time - start_time,
                    "status_code": 500,
                    "success": False,
                    "error": str(e)
                })
            return {"success": False, "latency": end_time - start_time, "error": str(e)}

    def run_load_test(self, 
                     queries: List[str],
                     rate: float,
                     duration: int = 60,
                     max_workers: int = 10) -> Dict[str, Any]:
        """
        Run load test with specified request rate.
        
        Args:
            queries: List of queries to use
            rate: Requests per second
            duration: Test duration in seconds
            max_workers: Maximum number of concurrent threads
        """
        print(f"\nStarting load test at {rate} RPS for {duration} seconds...")
        
        start_time = time.time()
        request_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while time.time() - start_time < duration:
                # Calculate sleep time to maintain rate
                next_request_time = start_time + (request_count + 1) / rate
                sleep_time = next_request_time - time.time()
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Submit request
                query = queries[request_count % len(queries)]
                executor.submit(self.send_request, query)
                request_count += 1
        
        # Calculate metrics
        latencies = [r["latency"] for r in self.results]
        success_rate = (self.success_count / (self.success_count + self.error_count)) * 100
        
        metrics = {
            "total_requests": len(self.results),
            "success_rate": success_rate,
            "avg_latency": np.mean(latencies),
            "p50_latency": np.percentile(latencies, 50),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "min_latency": min(latencies),
            "max_latency": max(latencies)
        }
        
        return metrics

    def plot_results(self, save_path: str = "load_test_results.png"):
        """Plot test results."""
        timestamps = [datetime.fromisoformat(r["timestamp"]) for r in self.results]
        latencies = [r["latency"] for r in self.results]
        success = [r["success"] for r in self.results]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot latencies
        ax1.plot(timestamps, latencies, 'b-', label='Latency')
        ax1.set_ylabel('Latency (seconds)')
        ax1.set_title('Request Latencies Over Time')
        ax1.grid(True)
        
        # Plot success/failure
        ax2.scatter(timestamps, success, c=['g' if s else 'r' for s in success], 
                   label='Success/Failure')
        ax2.set_ylabel('Success (1) / Failure (0)')
        ax2.set_xlabel('Time')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"\nResults plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Load test the RAG service')
    parser.add_argument('--rate', type=float, default=1.0,
                      help='Request rate (requests per second)')
    parser.add_argument('--duration', type=int, default=60,
                      help='Test duration in seconds')
    parser.add_argument('--max-workers', type=int, default=10,
                      help='Maximum number of concurrent workers')
    args = parser.parse_args()
    
    # Test queries
    test_queries = [
        "How do hummingbirds move?",
        "Tell me about cats",
        "What do dogs do?",
        "Can birds fly?",
        "What are domestic pets?",
        "Tell me about mammals"
    ]
    
    # Run load test
    tester = LoadTester()
    metrics = tester.run_load_test(
        queries=test_queries,
        rate=args.rate,
        duration=args.duration,
        max_workers=args.max_workers
    )
    
    # Print results
    print("\n=== Load Test Results ===")
    print(f"Total Requests: {metrics['total_requests']}")
    print(f"Success Rate: {metrics['success_rate']:.2f}%")
    print(f"Average Latency: {metrics['avg_latency']:.2f}s")
    print(f"P50 Latency: {metrics['p50_latency']:.2f}s")
    print(f"P95 Latency: {metrics['p95_latency']:.2f}s")
    print(f"P99 Latency: {metrics['p99_latency']:.2f}s")
    print(f"Min Latency: {metrics['min_latency']:.2f}s")
    print(f"Max Latency: {metrics['max_latency']:.2f}s")
    
    # Plot results
    tester.plot_results()
    
    # Save detailed results to JSON
    with open('load_test_results.json', 'w') as f:
        json.dump({
            "metrics": metrics,
            "detailed_results": tester.results
        }, f, indent=2)

if __name__ == "__main__":
    main() 