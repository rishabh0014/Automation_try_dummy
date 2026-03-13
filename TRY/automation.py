import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


df = pd.read_excel("error_dataset_with_testcases_2500.xlsx")

errors = df["error_message"].tolist()
row_ids = df["row_number"].tolist()
testcase = df["testcase"].tolist()


def normalize_error(msg):
    msg = msg.lower()
    msg = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', msg)
    msg = re.sub(r'\d+', '<NUM>', msg)
    msg = re.sub(r'\b[a-f0-9]{8,}\b', '<HASH>', msg)
    return msg


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


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


processed_text = []

for err, tc in zip(errors, testcase):
    normalized = normalize_error(err)
    pattern = extract_pattern(normalized)
    combined = tc + " " + normalized + " " + pattern
    processed_text.append(combined)


embeddings = model.encode(
    processed_text,
    convert_to_numpy=True,
    show_progress_bar=True
)

embeddings = embeddings.astype("float32")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)


def search_similar_error(query, testcase):
    normalized = normalize_error(query)
    pattern = extract_pattern(normalized)
    combined = testcase + " " + normalized + " " + pattern
    query_embedding = model.encode([combined])
    query_embedding = query_embedding.astype("float32")
    faiss.normalize_L2(query_embedding)
    k = 5

    distances, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(distances[0], indices[0]):
        if score > 0.75:
            results.append({
                "row_number": row_ids[idx],
                "error": errors[idx],
                "similarity": float(score)
            })
    return results


query = "Error: Device 999 timeout after 1200 ms"

results = search_similar_error(query, "pcie_reset_test")

for r in results:
    print(r)