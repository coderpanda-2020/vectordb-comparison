import os
import time
import psutil
import numpy as np
import pandas as pd
from memory_profiler import memory_usage
import faiss
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import chromadb

# Benchmark parameters
DIMENSIONS = 128
VOLUMES = [10_000, 100_000, 1_000_000]
K = 10  # Number of nearest neighbors to retrieve
NUM_QUERIES = 100

def get_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def generate_data(num_vectors, dim):
    np.random.seed(42)
    return np.random.random((num_vectors, dim)).astype('float32')

def benchmark_faiss(vectors, queries):
    print("Testing FAISS...")
    start_mem = get_memory()
    start_time = time.time()
    
    # We use IVF for a more realistic comparison with other DBs, or Flat for exact
    # For large scale, we should probably just use Flat to simplify and get exact recall baseline, 
    # but HNSW is what Qdrant/Chroma use mostly. Let's use HNSW.
    index = faiss.IndexHNSWFlat(DIMENSIONS, 32)
    index.add(vectors)
    
    index_time = time.time() - start_time
    mem_used = max(0, get_memory() - start_mem)
    
    start_time = time.time()
    _, I = index.search(queries, K)
    query_time = (time.time() - start_time) / NUM_QUERIES
    
    return index_time, query_time, mem_used, I

def benchmark_qdrant(vectors, queries):
    print("Testing Qdrant...")
    client = QdrantClient(":memory:")
    collection_name = "test_collection"
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=DIMENSIONS, distance=Distance.COSINE),
    )
    
    # Qdrant client expects Python lists/dicts
    points = [
        PointStruct(id=i, vector=vectors[i].tolist())
        for i in range(len(vectors))
    ]
    
    start_mem = get_memory()
    start_time = time.time()
    
    # Batch upload to avoid memory explosion in payload creation
    batch_size = 10_000
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=points[i:i+batch_size]
        )
        
    index_time = time.time() - start_time
    mem_used = max(0, get_memory() - start_mem)
    
    start_time = time.time()
    results = []
    for q in queries:
        hits = client.query_points(
            collection_name=collection_name,
            query=q.tolist(),
            limit=K
        )
        results.append([hit.id for hit in hits.points])
    query_time = (time.time() - start_time) / NUM_QUERIES
    
    return index_time, query_time, mem_used, np.array(results)

def benchmark_chroma(vectors, queries):
    print("Testing Chroma...")
    client = chromadb.Client()
    collection = client.create_collection(name="test_collection")
    
    ids = [str(i) for i in range(len(vectors))]
    
    start_mem = get_memory()
    start_time = time.time()
    
    batch_size = 5_000
    for i in range(0, len(vectors), batch_size):
        collection.add(
            embeddings=vectors[i:i+batch_size].tolist(),
            ids=ids[i:i+batch_size]
        )
        
    index_time = time.time() - start_time
    mem_used = max(0, get_memory() - start_mem)
    
    start_time = time.time()
    results = collection.query(
        query_embeddings=queries.tolist(),
        n_results=K
    )
    query_time = (time.time() - start_time) / NUM_QUERIES
    
    # Chroma returns ids as strings, convert to int for comparison
    parsed_results = [[int(id) for id in row] for row in results['ids']]
    
    # Clean up client
    client.delete_collection("test_collection")
    
    return index_time, query_time, mem_used, np.array(parsed_results)

def calculate_recall(ground_truth, predictions):
    recalls = []
    for gt, pred in zip(ground_truth, predictions):
        hits = set(gt).intersection(set(pred))
        recalls.append(len(hits) / K)
    return np.mean(recalls)

def run_benchmarks():
    results = []
    
    for volume in VOLUMES:
        print(f"\n--- Running benchmark for {volume} vectors ---")
        vectors = generate_data(volume, DIMENSIONS)
        queries = generate_data(NUM_QUERIES, DIMENSIONS)
        
        # Get ground truth using FAISS Flat L2 for exact search
        print("Computing exact nearest neighbors for recall baseline...")
        exact_index = faiss.IndexFlatL2(DIMENSIONS)
        exact_index.add(vectors)
        _, ground_truth = exact_index.search(queries, K)
        
        # FAISS
        f_idx_time, f_q_time, f_mem, f_res = benchmark_faiss(vectors, queries)
        f_recall = calculate_recall(ground_truth, f_res)
        results.append({"DB": "FAISS", "Volume": volume, "Index Time (s)": f_idx_time, "Query Latency (s)": f_q_time, "Memory (MB)": f_mem, "Recall": f_recall})
        
        # Qdrant
        q_idx_time, q_q_time, q_mem, q_res = benchmark_qdrant(vectors, queries)
        q_recall = calculate_recall(ground_truth, q_res)
        results.append({"DB": "Qdrant", "Volume": volume, "Index Time (s)": q_idx_time, "Query Latency (s)": q_q_time, "Memory (MB)": q_mem, "Recall": q_recall})
        
        # Chroma
        c_idx_time, c_q_time, c_mem, c_res = benchmark_chroma(vectors, queries)
        c_recall = calculate_recall(ground_truth, c_res)
        results.append({"DB": "Chroma", "Volume": volume, "Index Time (s)": c_idx_time, "Query Latency (s)": c_q_time, "Memory (MB)": c_mem, "Recall": c_recall})

    df = pd.DataFrame(results)
    df.to_csv("results/benchmark_results.csv", index=False)
    print("\nBenchmark complete! Results saved to results/benchmark_results.csv")
    print(df)

if __name__ == "__main__":
    run_benchmarks()
