import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI()

# Example documents in memory
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

# Set up devices
llm_device = "mps" if torch.backends.mps.is_available() else "cpu"
# Keep embedding model on CPU due to MPS limitations with some operations
embed_device = "cpu"
print(f"Using devices - LLM: {llm_device}, Embedding: {embed_device}")

# 1. Load embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(embed_device)

# Basic Chat LLM
chat_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device=llm_device)
# Note: try this 1.5B model if you got enough GPU memory
# chat_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")

def get_embedding(text: str) -> np.ndarray:
    """Compute a simple average-pool embedding."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    # Move inputs to CPU
    inputs = {k: v.to(embed_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Precompute document embeddings
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])

def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """Retrieve top-k docs via dot-product similarity."""
    sims = doc_embeddings @ query_emb.T
    sims = sims.ravel()  # Flatten the array properly
    top_k_indices = np.argsort(sims)[::-1][:k]
    # Fix numpy deprecation warning by accessing individual elements properly
    return [(documents[int(i)], float(sims[int(i)])) for i in top_k_indices]

def rag_pipeline(query: str, k: int = 2) -> str:
    # Step 1: Input embedding
    query_emb = get_embedding(query)
    
    # Step 2: Retrieval with similarity scores
    retrieved_docs_with_scores = retrieve_top_k(query_emb, k)
    
    # Filter out low-similarity documents (threshold can be adjusted)
    relevant_docs = [doc for doc, score in retrieved_docs_with_scores if score > 0.1]
    
    if not relevant_docs:
        return "I don't have enough relevant information to answer this question accurately."
    
    # Construct the prompt from query + retrieved docs
    context = "\n".join(f"- {doc}" for doc in relevant_docs)
    prompt = (
        "System: You are a direct and concise assistant. Provide only short, factual answers.\n\n"
        f"Context:\n{context}\n\n"
        f"Human: {query}\n"
        "Assistant: Give a one-sentence answer using only the context provided."
    )
    
    # Step 3: LLM Output
    response = chat_pipeline(
        prompt, 
        max_new_tokens=30,  # Using max_new_tokens instead of max_length
        do_sample=True,
        temperature=0.1,  # Reduced temperature for more deterministic responses
        num_return_sequences=1,
        truncation=True,
        pad_token_id=chat_pipeline.tokenizer.eos_token_id,
        eos_token_id=chat_pipeline.tokenizer.eos_token_id,
        return_full_text=False  # Only return the generated text, not the prompt
    )[0]["generated_text"]
    
    # Clean up the response
    try:
        # Get everything after the last "Assistant:" if it exists
        if "Assistant:" in response:
            answer = response.split("Assistant:")[-1]
        else:
            answer = response
            
        # Clean up the answer
        answer = answer.strip()
        answer = answer.split("\n")[0]  # Take only the first line
        answer = answer.split(".")[0] + "."  # Take only the first sentence and add period
        
        # Remove any meta-text patterns
        patterns_to_remove = [
            "Based on the context,",
            "According to the context,",
            "The context states that",
            "Therefore,",
            "To answer your question,",
            "I can tell you that",
            "The answer is"
        ]
        
        for pattern in patterns_to_remove:
            answer = answer.replace(pattern, "").strip()
        
        return answer if answer else "No clear answer found in the context."
    except:
        return "Error processing the response."

# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 2

@app.post("/rag")
async def predict(payload: QueryRequest):
    try:
        result = rag_pipeline(payload.query, payload.k)
        return {
            "query": payload.query,
            "result": result,
            "status": "complete"
        }
    except Exception as e:
        return {
            "query": payload.query,
            "result": str(e),
            "status": "error"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 