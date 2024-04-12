# FFVec

> A blazing fast (currently) in memory vector database.

## Features

- Advanced querying using the logistic regression method described [here](https://x.com/karpathy/status/1647278857601564672).
- Binary embeddings
- Metadata filtering (for now only support equality matching)
- Written in less than 1000 lines of pure C
- Built in Python bindings
- Much faster writes than FAISS with HNSW
- Hackable and extendable if you so desire

## Benchmarks
> run these yourself with bench.py

```
Benchmarking with 100,000 documents
FFVec Query Average Time: 1.64ms
FAISS Flat Query Average Time: 10.56ms
FAISS HNSW Query Average Time: 1.55ms
FAISS IVFPQ Query Average Time: 0.18ms


Benchmarking with 500,000 documents

```

## Roadmap
- [] On disk indexing
- [] More efficient non-vector querying
- [] More non-vector querying options (ne, range, etc)
