import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi


# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_excel("error_dataset_with_testcases_2500.xlsx")

errors = df["error_message"].tolist()
row_ids = df["row_number"].tolist()
testcases = df["testcase"].tolist()


# -----------------------------
# TEMPLATE EXTRACTION
# -----------------------------

def extract_template(msg):

    msg = msg.lower()

    msg = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', msg)
    msg = re.sub(r'\d+', '<NUM>', msg)
    msg = re.sub(r'\b[a-f0-9]{8,}\b', '<HASH>', msg)

    return msg


# -----------------------------
# PATTERN EXTRACTION
# -----------------------------

def extract_pattern(msg):

    patterns = [
        r"timeout",
        r"link training failed",
        r"reset failed",
        r"dma transfer error",
        r"queue overflow",
        r"firmware assertion",
        r"memory read failure",
        r"invalid namespace",
        r"temperature threshold",
        r"power state transition",
        r"interrupt .* not handled"
    ]

    detected = []

    for p in patterns:
        if re.search(p, msg):
            detected.append(p)

    return " ".join(detected)


# -----------------------------
# MODEL
# -----------------------------

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# -----------------------------
# PREPROCESS DATA
# -----------------------------

processed_text = []

for err, tc in zip(errors, testcases):

    template = extract_template(err)

    pattern = extract_pattern(template)

    combined = tc + " " + template + " " + pattern

    processed_text.append(combined)


# -----------------------------
# EMBEDDINGS
# -----------------------------

embeddings = model.encode(
    processed_text,
    convert_to_numpy=True,
    show_progress_bar=True
)

embeddings = embeddings.astype("float32")

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# -----------------------------
# BM25 KEYWORD INDEX
# -----------------------------

tokenized_corpus = [text.split() for text in processed_text]

bm25 = BM25Okapi(tokenized_corpus)


# -----------------------------
# HYBRID SEARCH
# -----------------------------

def hybrid_search(query, testcase):

    template = extract_template(query)

    pattern = extract_pattern(template)

    combined = testcase + " " + template + " " + pattern

    # -------- semantic search --------

    query_embedding = model.encode([combined]).astype("float32")

    faiss.normalize_L2(query_embedding)

    k = 50

    distances, indices = index.search(query_embedding, k)


    # -------- keyword search --------

    tokenized_query = combined.split()

    bm25_scores = bm25.get_scores(tokenized_query)


    results = []

    for score, idx in zip(distances[0], indices[0]):

        embedding_score = score

        keyword_score = bm25_scores[idx] / 10

        final_score = 0.7 * embedding_score + 0.3 * keyword_score

        if final_score > 0.7:

            results.append({
                "row_number": row_ids[idx],
                "error": errors[idx],
                "score": float(final_score)
            })


    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:5]


# -----------------------------
# TEST
# -----------------------------

query = "Error: Device 999 timeout after 1200 ms"

results = hybrid_search(query, "pcie_reset_test")

for r in results:
    print(r)