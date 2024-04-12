# FFVec

> A blazing fast (currently) in memory vector database.

## Features

- Advanced querying using the logistic classifier method described [here](https://x.com/karpathy/status/1647278857601564672).
- Binary embeddings
- Metadata filtering (for now only support equality matching)
- Written in less than 1000 lines of pure C
- Built in Python bindings
- Much faster writes than FAISS with HNSW
- Hackable and extendable if you so desire

## Benchmarks
> run these yourself with bench.py

Here the FFVec queries are the advanced version that trains a logistic classifier in the process of doing the query.

```
Benchmarking with 100,000 documents

FFVec Query Average Time: 1.64ms
FAISS Flat Query Average Time: 10.56ms
FAISS HNSW Query Average Time: 1.55ms
FAISS IVFPQ Query Average Time: 0.18ms


Benchmarking with 500,000 documents

FFVec Query Average Time: 2.98ms
FAISS Flat Query Average Time: 52.11ms
FAISS HNSW Query Average Time: 2.28ms
FAISS IVFPQ Query Average Time: 0.73ms


Benchmarking with 1,000,000 documents

FFVec Query Average Time: 4.96ms
FAISS Flat Query Average Time: 104.27ms
FAISS HNSW Query Average Time: 0.93ms
FAISS IVFPQ Query Average Time: 1.33ms

```

## Roadmap
- [] On disk indexing
- [] More efficient non-vector querying
- [] More non-vector querying options (ne, range, etc)
