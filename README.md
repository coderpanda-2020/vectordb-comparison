# Vector Database Benchmarks: FAISS vs Qdrant vs Chroma

This repository contains scripts to benchmark three popular vector databases: FAISS, Qdrant, and Chroma. We evaluate them across 4 core metrics (Indexing Time, Query Latency, Memory Usage, and Recall) on various data volumes (10k, 100k, 1M vectors).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/coderpanda-2020/vectordb-comparison.git
cd vectordb-comparison
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note for Docker Users:**
> For the massive 1,000,000 vector scale, we highly recommend running Qdrant via Docker rather than local memory to avoid hours of slow Python ingestion:
> ```bash
> docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
> ```
> Then modify `client = QdrantClient(":memory:")` to `client = QdrantClient("http://localhost:6333")` in `benchmark.py`.

> **Note for GPU Users:**
> If you have an NVIDIA GPU, you can swap `faiss-cpu` for `faiss-gpu` in `requirements.txt` to significantly accelerate indexing and querying.

## Running the Benchmark

Simply run the benchmark script from the root folder:
```bash
python src/benchmark.py
```
This will generate a `benchmark_results.csv` file in the `results/` folder.

## Visualizing the Results

If you want to visualize the results using matplotlib and seaborn, run:
```bash
python src/visualize.py
```
*(Alternatively, you can open `docs/visualizations.html` to view an interactive web dashboard!)*

## Running Tests

To ensure the core benchmark utilities (like recall calculation) are working:
```bash
pytest tests/
```
