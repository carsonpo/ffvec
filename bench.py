import ffvec
import faiss
import numpy as np
import time
from tqdm.auto import tqdm


np.random.seed(0)
dimension = 384
num_docs = 100000
embeddings = np.random.rand(num_docs, dimension).astype("float32")

# Initialize ffvec
vs = ffvec.VectorSet()
metadata = [{"genre": "sci-fi", "year": 1977}] * num_docs
for i in range(num_docs):
    vs.add(embeddings[i].tolist(), metadata[i])

# Initialize FAISS Flat index
index_flat = faiss.IndexFlatL2(dimension)
index_flat.add(embeddings)


# Initialize FAISS HNSW index
index_hnsw = faiss.IndexHNSWFlat(dimension, 32)  # M = 32 by default
index_hnsw.hnsw.efConstruction = 200  # Can be tuned for better accuracy
# index_hnsw.add(embeddings)

for i in tqdm(range(0, num_docs, 1000)):
    index_hnsw.add(embeddings[i : i + 1000])


nlist = 100  # Number of clusters
m = 8  # The number of sub-vector quantizers
quantizer = faiss.IndexFlatL2(dimension)  # the quantizer
index_ivfpq = faiss.IndexIVFPQ(
    quantizer, dimension, nlist, m, 8
)  # 8 bits per sub-vector
index_ivfpq.train(embeddings)  # Training on the dataset
index_ivfpq.add(embeddings)  # Add embeddings to the index

# Prepare query vector (also simulated)
query_vec = np.random.rand(1, dimension).astype("float32")

# Benchmark ffvec
start_ffvec = time.perf_counter()
for _ in range(100):
    vs_out = vs.query_with_metadata(query_vec[0].tolist(), {"genre": "sci-fi"}, 2)
time_ffvec = (time.perf_counter() - start_ffvec) / 100.0 * 1000

# Benchmark FAISS Flat
start_faiss_flat = time.perf_counter()
for _ in range(100):
    D_flat, I_flat = index_flat.search(query_vec, 2)
time_faiss_flat = (time.perf_counter() - start_faiss_flat) / 100.0 * 1000

# Benchmark FAISS HNSW
start_faiss_hnsw = time.perf_counter()
for _ in range(100):
    D_hnsw, I_hnsw = index_hnsw.search(query_vec, 2)
time_faiss_hnsw = (time.perf_counter() - start_faiss_hnsw) / 100.0 * 1000


# Benchmark FAISS IVFPQ
start_faiss_ivfpq = time.perf_counter()
for _ in range(100):
    index_ivfpq.nprobe = 10  # Number of clusters to visit, can be tuned
    D_ivfpq, I_ivfpq = index_ivfpq.search(query_vec, 2)
time_faiss_ivfpq = (time.perf_counter() - start_faiss_ivfpq) / 100.0 * 1000

# Benchmark ffvec
start_ffvec = time.perf_counter()
for _ in range(100):
    vs_out = vs.query_with_metadata(query_vec[0].tolist(), {"genre": "sci-fi"}, 2)
time_ffvec = (time.perf_counter() - start_ffvec) / 100.0 * 1000

# Benchmark FAISS Flat
start_faiss_flat = time.perf_counter()
for _ in range(100):
    D_flat, I_flat = index_flat.search(query_vec, 2)
time_faiss_flat = (time.perf_counter() - start_faiss_flat) / 100.0 * 1000

# Benchmark FAISS HNSW
start_faiss_hnsw = time.perf_counter()
for _ in range(100):
    D_hnsw, I_hnsw = index_hnsw.search(query_vec, 2)
time_faiss_hnsw = (time.perf_counter() - start_faiss_hnsw) / 100.0 * 1000


# Benchmark FAISS IVFPQ
start_faiss_ivfpq = time.perf_counter()
for _ in range(100):
    index_ivfpq.nprobe = 10  # Number of clusters to visit, can be tuned
    D_ivfpq, I_ivfpq = index_ivfpq.search(query_vec, 2)
time_faiss_ivfpq = (time.perf_counter() - start_faiss_ivfpq) / 100.0 * 1000

# Print results
print(f"FFVec Query Average Time: {time_ffvec:.2f}ms")
print(f"FAISS Flat Query Average Time: {time_faiss_flat:.2f}ms")
print(f"FAISS HNSW Query Average Time: {time_faiss_hnsw:.2f}ms")
print(f"FAISS IVFPQ Query Average Time: {time_faiss_ivfpq:.2f}ms")
