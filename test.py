import ffvec
import openai
import time


vs = ffvec.VectorSet()
docs = [
    "a long time ago in a galaxy far, far away",
    "the cow goes moo",
    "the cat goes meow",
    "why is the sky blue",
    "how many ping pong balls can fit in a 747",
]
vec = (
    openai.embeddings.create(
        input=docs[0], model="text-embedding-3-small", dimensions=384
    )
    .data[0]
    .embedding
)
query_vec = list(
    openai.embeddings.create(
        input="I think many ping pong balls would definitely fit in a 747",
        model="text-embedding-3-small",
        dimensions=384,
    )
    .data[0]
    .embedding
)

e = (
    openai.embeddings.create(
        input=docs[0], model="text-embedding-3-small", dimensions=384
    )
    .data[0]
    .embedding
)
# for _ in range(1_000_000):
#     vs.add(e, {"genre": "sci-fi", "year": 1977})

for doc in docs:
    vec = (
        openai.embeddings.create(
            input=doc, model="text-embedding-3-small", dimensions=384
        )
        .data[0]
        .embedding
    )
    vs.add(vec, {"genre": "sci-fi", "year": 1977, 'text': doc})

start = time.perf_counter()
for _ in range(100):
    out = vs.advanced_query(
        query_vec,
        {"genre": "sci-fi"},
        2,
    )


print(f"Query took {((time.perf_counter() - start) / 100.0 * 1000):.2f}ms")
print(out)
