from typing import List, Union
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import pickle
import os
import cupy as cp
import numpy as np
import csv
import time

from task import compare_ann_recall_with_cupy


llm_device="cuda:0"
embed_device="cuda:0"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(embed_device)


def get_embedding(texts: Union[str, List[str]]) -> cp.ndarray:
    try:
        if isinstance(texts, str):
            texts = [texts]

        inputs = embed_tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(embed_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = embed_model(**inputs)
        # Pooling: mean of last_hidden_state for each sequence
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return cp.asarray(embeddings)  # shape: (batch_size, hidden_dim)
    except Exception as e:
        print(f"Embedding error: {e}")
        return cp.zeros((len(texts), embed_model.config.hidden_size))
    


EMBEDDINGS_CACHE_FILE = "emb_cache"
DOCUMENTS_CACHE_FILE = "doc_cache"


def load_or_create_embeddings():
    """Load cached embeddings if they exist, otherwise create and cache them."""
    global documents, document_answers, doc_embeddings
    
    if os.path.exists(EMBEDDINGS_CACHE_FILE) and os.path.exists(DOCUMENTS_CACHE_FILE):
        print("Loading cached embeddings and documents...")
        try:
            with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
                doc_embeddings = pickle.load(f)
            with open(DOCUMENTS_CACHE_FILE, 'rb') as f:
                cached_data = pickle.load(f)
                documents = cached_data['documents']
                document_answers = cached_data['document_answers']
            
            print(f"Successfully loaded cached embeddings and {len(documents)} documents")
            
            # Verify the cache is valid
            if len(documents) == 0 or doc_embeddings.shape[0] != len(documents):
                print("Cache appears to be invalid, recreating...")
                os.remove(EMBEDDINGS_CACHE_FILE)
                os.remove(DOCUMENTS_CACHE_FILE)
                return load_or_create_embeddings()
                
            return documents, document_answers, doc_embeddings
        except Exception as e:
            print(f"Error loading cached embeddings: {e}")
            print("Recreating cache...")
            if os.path.exists(EMBEDDINGS_CACHE_FILE):
                os.remove(EMBEDDINGS_CACHE_FILE)
            if os.path.exists(DOCUMENTS_CACHE_FILE):
                os.remove(DOCUMENTS_CACHE_FILE)

    print("Creating new embeddings...")
    # Load BioASQ dataset
    print("Loading BioASQ dataset...")
    dataset = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages")
    qa_data = dataset['test']

    # Create documents from the dataset
    documents = []
    document_answers = {}
    for item in qa_data:
        documents.append(item['answer'])
        document_answers[item['answer']] = item['question']

    print(f"Loaded {len(documents)} documents from BioASQ dataset")
    
    # Compute embeddings in batches to avoid memory issues
    print("Computing document embeddings...")
    batch_size = 32
    doc_embeddings_list = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_embeddings = [get_embedding(doc) for doc in batch]
        doc_embeddings_list.extend(batch_embeddings)
        print(f"Processed {i + len(batch)}/{len(documents)} documents")
    
    doc_embeddings = cp.vstack(doc_embeddings_list)
    print("Document embeddings completed")

    # Cache the results
    print("Caching embeddings and documents...")
    try:
        with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
            pickle.dump(doc_embeddings, f)
        with open(DOCUMENTS_CACHE_FILE, 'wb') as f:
            pickle.dump({
                'documents': documents,
                'document_answers': document_answers
            }, f)
        print(f"Successfully cached embeddings and {len(documents)} documents")
    except Exception as e:
        print(f"Error caching embeddings: {e}")

    return documents, document_answers, doc_embeddings


if __name__ == "__main__":
    # documents, document_answers, doc_embeddings = load_or_create_embeddings()
    # # print(documents[0])
    # # print(document_answers[documents[0]])

    # print(doc_embeddings.shape)
    # #
    # A = doc_embeddings

    # K = 1
    # n_clusters = 1
    # queries = get_embedding(documents[0:50])
    # N, D = A.shape[0], A.shape[1]

    # batch_sizes = [2**i for i in range(1, 30)]
    # for batch in batch_sizes:
    #     start = time.time()
    #     compare_ann_recall_with_cupy(N, D, A, queries, K, num_clusters=n_clusters)
    #     end = time.time()

    #     print(f"{batch} time: {end-start}")
    documents, document_answers, doc_embeddings = load_or_create_embeddings()
    # print(documents[0])
    # print(document_answers[documents[0]])

    print(doc_embeddings.shape)
    #
    A = doc_embeddings

    K_values = [1, 3, 5, 10, 15, 20]  # Different values of K (number of nearest neighbors)
    n_clusters_values = [1, 3, 5, 8, 10, 20, 30, 40, 50, 100, 200]


    queries = get_embedding(documents[0:50])
    N, D = A.shape[0], A.shape[1]

    recall_results = []
    for K in K_values:
        for num_clusters in n_clusters_values:
            print(f"Testing K={K}, num_clusters={num_clusters}...")

            # Compute average recall for the current combination of K and num_clusters
            start = time.time()
            average_recall = compare_ann_recall_with_cupy(N, D, A, queries, K, num_clusters)
            end = time.time()


            # Save the result for later analysis
            recall_results.append((K, num_clusters, average_recall, float(end-start)))
            print(f"  Average recall: {average_recall:.4f}")

    # compare_ann_recall_with_cupy(N, D, A, queries, K, num_clusters=n_clusters)

    # Convert results to a numpy array for easy manipulation and analysis
    recall_results = np.array(recall_results)

    # Save the results to a CSV file for later analysis
    csv_file = "knn_recall_vs_clusters.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "n_clusters", "Average Recall", "Runtime"])  # Header
        writer.writerows(recall_results)  # Write the rows of results

    print(f"\n✅ Benchmarking complete. Results saved to '{csv_file}'")