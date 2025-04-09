import torch
import time
from transformers import AutoTokenizer, AutoModel, pipeline
import numpy as np

def test_device_availability():
    print("\n=== Device Availability Test ===")
    print(f"MPS (Apple GPU) available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current PyTorch version: {torch.__version__}")
    
    # Get the current device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device

def test_model_device(model, name="Model"):
    print(f"\n=== {name} Device Test ===")
    try:
        print(f"Model device: {next(model.parameters()).device}")
        return True
    except Exception as e:
        print(f"Error checking model device: {e}")
        return False

def measure_inference_time(func, *args, num_runs=5):
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = func(*args)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time:.4f} seconds")
    return avg_time

def main():
    print("\n=== Starting GPU Test ===")
    device = test_device_availability()
    
    # Load models
    print("\n=== Loading Models ===")
    EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
    embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    
    # First test on CPU
    print("\n=== CPU Testing ===")
    embed_model_cpu = AutoModel.from_pretrained(EMBED_MODEL_NAME).to('cpu')
    chat_pipeline_cpu = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        device='cpu'
    )
    
    # Test embedding model on CPU
    print("\nTesting embedding model (CPU):")
    test_text = "This is a test sentence for measuring inference speed."
    
    def embedding_inference_cpu():
        inputs = embed_tokenizer(test_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = embed_model_cpu(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    cpu_embed_time = measure_inference_time(embedding_inference_cpu)
    
    # Test chat model on CPU
    print("\nTesting chat model (CPU):")
    test_prompt = "Write a one-sentence story."
    
    def chat_inference_cpu():
        return chat_pipeline_cpu(
            test_prompt,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.1
        )
    
    cpu_chat_time = measure_inference_time(chat_inference_cpu)
    
    # Now test on MPS
    print("\n=== MPS Testing ===")
    embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device)
    chat_pipeline = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        device=device
    )
    
    # Test model devices
    test_model_device(embed_model, "Embedding Model")
    
    # Test embedding model on MPS
    print("\nTesting embedding model (MPS):")
    
    def embedding_inference_mps():
        inputs = embed_tokenizer(test_text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = embed_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    mps_embed_time = measure_inference_time(embedding_inference_mps)
    
    # Test chat model on MPS
    print("\nTesting chat model (MPS):")
    
    def chat_inference_mps():
        return chat_pipeline(
            test_prompt,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.1
        )
    
    mps_chat_time = measure_inference_time(chat_inference_mps)
    
    # Print speedup comparison
    print("\n=== Performance Comparison ===")
    embed_speedup = cpu_embed_time / mps_embed_time
    chat_speedup = cpu_chat_time / mps_chat_time
    print(f"Embedding model speedup (MPS vs CPU): {embed_speedup:.2f}x")
    print(f"Chat model speedup (MPS vs CPU): {chat_speedup:.2f}x")
    
    print("\n=== Memory Usage ===")
    if device == "mps":
        print("MPS is being utilized for computation")
    elif device == "cuda":
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    else:
        print("Running on CPU, no GPU memory to report")

if __name__ == "__main__":
    main() 