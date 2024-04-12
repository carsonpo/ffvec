import ffvec
import faiss
import numpy as np
import time
from tqdm.auto import tqdm

def benchmark_indices(num_docs):
    np.random.seed(0)
    dimension = 384
    embeddings = np.random.rand(num_docs, dimension).astype('float32')

    # FFVec Index
    vs = ffvec.VectorSet()
    metadata = [{}] * num_docs
    for i in range(num_docs):
        vs.add(embeddings[i].tolist(), metadata[i])

    # FAISS Flat Index
    index_flat = faiss.IndexFlatL2(dimension)
    index_flat.add(embeddings)

    # FAISS HNSW Index
    index_hnsw = faiss.IndexHNSWFlat(dimension, 32)
    index_hnsw.hnsw.efConstruction = 200
    

    for i in tqdm(range(0, num_docs, 10_000)):
        index_hnsw.add(embeddings[i:i+10_000])


    # FAISS IVFPQ Index
    nlist = 100
    m = 8
    quantizer = faiss.IndexFlatL2(dimension)
    index_ivfpq = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
    index_ivfpq.train(embeddings[0:10_000])
    # index_ivfpq.add(embeddings)

    for i in tqdm(range(0, num_docs, 10_000)):
        index_ivfpq.add(embeddings[i:i+10_000])

    # Prepare query vector
    query_vec = np.random.rand(1, dimension).astype('float32')

    for index_type, index in [('ffvec', vs), ('FAISS Flat', index_flat), ('FAISS HNSW', index_hnsw), ('FAISS IVFPQ', index_ivfpq)]:
        start_time = time.perf_counter()
        if index_type == 'ffvec':
            for _ in range(100):
                vs_out = vs.advanced_query(query_vec[0].tolist(), {}, 2)
        else:
            for _ in range(100):
                if index_type == 'FAISS IVFPQ':
                    index.nprobe = 10
                D, I = index.search(query_vec, 2)
        elapsed_time = (time.perf_counter() - start_time) / 100.0 * 1000

    # Benchmarking
    results = {}
    for index_type, index in [('ffvec', vs), ('FAISS Flat', index_flat), ('FAISS HNSW', index_hnsw), ('FAISS IVFPQ', index_ivfpq)]:
        start_time = time.perf_counter()
        if index_type == 'ffvec':
            for _ in range(100):
                vs_out = vs.advanced_query(query_vec[0].tolist(), {}, 2)
        else:
            for _ in range(100):
                if index_type == 'FAISS IVFPQ':
                    index.nprobe = 10
                D, I = index.search(query_vec, 2)
        elapsed_time = (time.perf_counter() - start_time) / 100.0 * 1000
        results[index_type] = elapsed_time

    return results

# Main loop to run benchmarks with different numbers of vectors
doc_sizes = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]  # Example sizes
for num_docs in doc_sizes:
    print(f"Benchmarking with {num_docs} documents")
    result = benchmark_indices(num_docs)
    for k, v in result.items():
        print(f"{k} Query Average Time: {v:.2f}ms")
