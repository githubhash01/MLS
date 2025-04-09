import subprocess
import time
import json
from datetime import datetime
import pandas as pd
import psutil
import os

# More conservative hyperparameter configurations
configs = [
    {"batch_size": 2, "wait_time": 1.0, "workers": 2},  # Baseline
    {"batch_size": 3, "wait_time": 1.5, "workers": 2},  # Slight increase
    {"batch_size": 4, "wait_time": 2.0, "workers": 3},  # Moderate
    {"batch_size": 3, "wait_time": 2.0, "workers": 3},  # Balance
]

def check_system_resources():
    """Check if system has enough resources to continue."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    print(f"\nSystem Status:")
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Memory Usage: {memory_percent}%")
    
    # If resources are too constrained, wait or exit
    if cpu_percent > 90 or memory_percent > 90:
        print("System resources too constrained. Waiting 30 seconds...")
        time.sleep(30)
        return False
    return True

def modify_server_config(config):
    """Modify the server configuration file with new parameters."""
    with open('task-2/serving_rag.py', 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if 'MAX_BATCH_SIZE =' in line:
            lines[i] = f"MAX_BATCH_SIZE = {config['batch_size']}  # Set by hyperparameter search\n"
        elif 'MAX_WAITING_TIME =' in line:
            lines[i] = f"MAX_WAITING_TIME = {config['wait_time']}  # Set by hyperparameter search\n"
        elif 'batch_thread_pool = ThreadPoolExecutor' in line:
            lines[i] = f"batch_thread_pool = ThreadPoolExecutor(max_workers={config['workers']})  # Set by hyperparameter search\n"
    
    with open('task-2/serving_rag.py', 'w') as f:
        f.writelines(lines)

def kill_servers():
    """Safely kill all RAG server processes."""
    try:
        subprocess.run(['pkill', '-f', 'serving_rag'])
    except:
        # If pkill fails, try to kill processes manually
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'] and 'serving_rag' in ' '.join(proc.info['cmdline']):
                    os.kill(proc.info['pid'], 9)
            except:
                continue
    time.sleep(5)

def run_test(config):
    """Run a single test with given configuration."""
    print(f"\nTesting configuration: {config}")
    
    # Check system resources before starting
    if not check_system_resources():
        return None
    
    # Kill any existing servers
    kill_servers()
    
    try:
        # Modify server configuration
        modify_server_config(config)
        
        # Start servers
        subprocess.Popen(['python', 'task-2/serving_rag_original.py'])
        subprocess.Popen(['python', 'task-2/serving_rag.py'])
        
        # Wait for servers to start and stabilize
        print("Waiting for servers to start...")
        time.sleep(15)
        
        # Check resources again before testing
        if not check_system_resources():
            kill_servers()
            return None
        
        # Run comparison test with shorter duration
        subprocess.run(['python', 'task-2/compare_performance.py', 
                       '--rate', '1.0', 
                       '--duration', '30',  # Reduced from 60 to 30 seconds
                       '--max-workers', '10'])
        
    except Exception as e:
        print(f"Error during test: {e}")
        kill_servers()
        return None
    
    finally:
        # Always try to kill servers
        kill_servers()
    
    # Cool down period between tests
    print("Cooling down for 20 seconds...")
    time.sleep(20)
    
    # Load and return results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        with open(f'comparison_results_{timestamp}.json', 'r') as f:
            results = json.load(f)
        return {
            **config,
            'original_latency': results['original_metrics']['avg_latency'],
            'optimized_latency': results['optimized_metrics']['avg_latency'],
            'original_success_rate': results['original_metrics']['success_rate'],
            'optimized_success_rate': results['optimized_metrics']['success_rate'],
            'latency_improvement': results['improvements']['latency_improvement_percent'],
            'throughput_improvement': results['improvements']['throughput_improvement_percent']
        }
    except:
        return None

def main():
    results = []
    for config in configs:
        result = run_test(config)
        if result:  # Only add successful results
            results.append(result)
        
    if not results:
        print("No successful tests completed. Try reducing system load and run again.")
        return
        
    # Convert results to DataFrame for analysis
    df = pd.DataFrame(results)
    print("\nHyperparameter Search Results:")
    print(df)
    
    # Save results
    df.to_csv('hyperparameter_search_results.csv', index=False)
    print("\nResults saved to hyperparameter_search_results.csv")
    
    # Find best configuration
    best_config = df.loc[df['latency_improvement'].idxmax()]
    print("\nBest Configuration:")
    print(f"Batch Size: {best_config['batch_size']}")
    print(f"Wait Time: {best_config['wait_time']}")
    print(f"Workers: {best_config['workers']}")
    print(f"Latency Improvement: {best_config['latency_improvement']:.2f}%")
    print(f"Success Rate: {best_config['optimized_success_rate']:.2f}%")

if __name__ == "__main__":
    main() 