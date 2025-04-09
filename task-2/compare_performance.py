import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from load_test import LoadTester
import requests
import time
import pandas as pd

class NonBatchedTester(LoadTester):
    """Tester for the non-batched version (direct processing)."""
    def __init__(self, base_url: str = "http://localhost:8001"):
        super().__init__(base_url)

def run_comparison_test(rate: float, duration: int, max_workers: int):
    """Run tests for both original and optimized versions."""
    test_queries = [
        "How do hummingbirds move?",
        "Tell me about cats",
        "What do dogs do?",
        "Can birds fly?",
        "What are domestic pets?",
        "Tell me about mammals"
    ]
    
    # Run test with original version (no batching)
    print("\n=== Testing Original Service (no batching) ===")
    original_tester = NonBatchedTester()
    original_metrics = original_tester.run_load_test(
        queries=test_queries,
        rate=rate,
        duration=duration,
        max_workers=max_workers
    )
    
    time.sleep(10)  # Wait between tests
    
    # Run test with optimized version (with batching)
    print("\n=== Testing Optimized Service (with batching) ===")
    optimized_tester = LoadTester()
    optimized_metrics = optimized_tester.run_load_test(
        queries=test_queries,
        rate=rate,
        duration=duration,
        max_workers=max_workers
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "test_parameters": {
            "rate": rate,
            "duration": duration,
            "max_workers": max_workers
        },
        "original_metrics": original_metrics,
        "optimized_metrics": optimized_metrics,
        "original_detailed_results": original_tester.results,
        "optimized_detailed_results": optimized_tester.results
    }
    
    # Calculate improvements
    latency_improvement = ((original_metrics["avg_latency"] - optimized_metrics["avg_latency"]) 
                         / original_metrics["avg_latency"] * 100)
    throughput_improvement = ((len(optimized_tester.results) - len(original_tester.results)) 
                            / len(original_tester.results) * 100)
    
    # Add improvements to results
    results["improvements"] = {
        "latency_improvement_percent": latency_improvement,
        "throughput_improvement_percent": throughput_improvement
    }
    
    # Save results
    with open(f'comparison_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create comparison plots
    plot_results(original_tester, optimized_tester)
    
    # Print summary
    print("\n=== Performance Comparison Summary ===")
    print(f"Original Average Latency: {original_metrics['avg_latency']:.2f}s")
    print(f"Optimized Average Latency: {optimized_metrics['avg_latency']:.2f}s")
    print(f"Latency Improvement: {latency_improvement:.1f}%")
    print(f"Original Throughput: {len(original_tester.results)/duration:.2f} RPS")
    print(f"Optimized Throughput: {len(optimized_tester.results)/duration:.2f} RPS")
    print(f"Throughput Improvement: {throughput_improvement:.1f}%")
    print(f"\nDetailed results saved to comparison_results_{timestamp}.json")

def plot_results(original_tester, optimized_tester):
    """Create separate plots for performance visualization."""
    # Set style with fallback
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
    
    # Plot 1: Request Latencies Over Time
    plt.figure(figsize=(10, 6))
    plt.title('Request Latencies Over Time')
    if original_tester.results:
        start_time = datetime.fromisoformat(original_tester.results[0]["timestamp"])
        times = [(datetime.fromisoformat(r["timestamp"]) - start_time).total_seconds() 
                for r in original_tester.results]
        plt.plot(times, [r['latency'] for r in original_tester.results], 
                'r-', label='Original', alpha=0.7)
    if optimized_tester.results:
        start_time = datetime.fromisoformat(optimized_tester.results[0]["timestamp"])
        times = [(datetime.fromisoformat(r["timestamp"]) - start_time).total_seconds() 
                for r in optimized_tester.results]
        plt.plot(times, [r['latency'] for r in optimized_tester.results], 
                'b-', label='Optimized', alpha=0.7)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Latency (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('latencies_over_time.png')
    plt.close()

    # Plot 2: Latency Distribution
    plt.figure(figsize=(10, 6))
    plt.title('Latency Distribution')
    if original_tester.results:
        plt.hist([r['latency'] for r in original_tester.results], 
                bins=20, alpha=0.5, label='Original', color='red')
    if optimized_tester.results:
        plt.hist([r['latency'] for r in optimized_tester.results], 
                bins=20, alpha=0.5, label='Optimized', color='blue')
    plt.xlabel('Latency (seconds)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig('latency_distribution.png')
    plt.close()

    # Plot 3: Cumulative Latency Distribution
    plt.figure(figsize=(10, 6))
    plt.title('Cumulative Latency Distribution')
    if original_tester.results:
        latencies = sorted([r['latency'] for r in original_tester.results])
        cumulative = np.arange(1, len(latencies) + 1) / len(latencies)
        plt.plot(latencies, cumulative, 'r-', label='Original', alpha=0.7)
    if optimized_tester.results:
        latencies = sorted([r['latency'] for r in optimized_tester.results])
        cumulative = np.arange(1, len(latencies) + 1) / len(latencies)
        plt.plot(latencies, cumulative, 'b-', label='Optimized', alpha=0.7)
    plt.xlabel('Latency (seconds)')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_latency.png')
    plt.close()

    # Plot 4: Success Rate Over Time
    plt.figure(figsize=(10, 6))
    plt.title('Success Rate Over Time')
    window = 10  # Rolling window size
    if original_tester.results:
        success = [1 if r['success'] else 0 for r in original_tester.results]
        rolling_success = pd.Series(success).rolling(window=window).mean()
        plt.plot(range(len(rolling_success)), rolling_success, 'r-', label='Original', alpha=0.7)
    if optimized_tester.results:
        success = [1 if r['success'] else 0 for r in optimized_tester.results]
        rolling_success = pd.Series(success).rolling(window=window).mean()
        plt.plot(range(len(rolling_success)), rolling_success, 'b-', label='Optimized', alpha=0.7)
    plt.xlabel('Request Number')
    plt.ylabel('Success Rate (rolling average)')
    plt.legend()
    plt.grid(True)
    plt.savefig('success_rate.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare RAG service performance')
    parser.add_argument('--rate', type=float, default=1.0,
                      help='Request rate (requests per second)')
    parser.add_argument('--duration', type=int, default=60,
                      help='Test duration in seconds')
    parser.add_argument('--max-workers', type=int, default=10,
                      help='Maximum number of concurrent workers')
    args = parser.parse_args()
    
    run_comparison_test(args.rate, args.duration, args.max_workers)

if __name__ == "__main__":
    main() 