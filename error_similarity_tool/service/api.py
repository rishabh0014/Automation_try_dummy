from fastapi import FastAPI
import faiss
import pickle
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

app = FastAPI()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index = faiss.read_index("model/faiss_index.bin")

bm25 = pickle.load(open("model/bm25.pkl", "rb"))
errors = pickle.load(open("model/errors.pkl", "rb"))
row_ids = pickle.load(open("model/rows.pkl", "rb"))
testcases = pickle.load(open("model/testcases.pkl", "rb"))


class Query(BaseModel):
    testcase: str
    error: str


def preprocess(msg):

    msg = msg.lower()

    msg = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', msg)
    msg = re.sub(r'\d+', '<NUM>', msg)

    return msg


@app.post("/search_error")
def search_error(query: Query):

    processed = query.testcase + " " + preprocess(query.error)

    embedding = model.encode([processed]).astype("float32")

    faiss.normalize_L2(embedding)

    k = 20

    distances, indices = index.search(embedding, k)

    tokenized_query = processed.split()

    bm25_scores = bm25.get_scores(tokenized_query)

    results = []

    for score, idx in zip(distances[0], indices[0]):

        embed_score = score

        keyword_score = bm25_scores[idx] / 10

        final_score = 0.7 * embed_score + 0.3 * keyword_score

        if final_score > 0.7:

            results.append({
                "row_number": row_ids[idx],
                "error": errors[idx],
                "score": float(final_score)
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return {"matches": results[:5]}