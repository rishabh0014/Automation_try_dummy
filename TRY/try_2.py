import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


df = pd.read_excel("error_dataset_with_testcases_2500.xlsx")

errors = df["error_message"].tolist()
row_ids = df["row_number"].tolist()
testcases = df["testcase"].tolist()


# ------------------------
# TEMPLATE EXTRACTION
# ------------------------

def extract_template(msg):

    msg = msg.lower()

    msg = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', msg)
    msg = re.sub(r'\d+', '<NUM>', msg)
    msg = re.sub(r'\b[a-f0-9]{8,}\b', '<HASH>', msg)

    return msg


# ------------------------
# PATTERN EXTRACTION
# ------------------------

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


# ------------------------
# LOAD MODEL
# ------------------------

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ------------------------
# PREPARE TEXT
# ------------------------

processed_text = []

for err, tc in zip(errors, testcases):

    template = extract_template(err)

    pattern = extract_pattern(template)

    combined = tc + " " + template + " " + pattern

    processed_text.append(combined)


# ------------------------
# EMBEDDINGS
# ------------------------

embeddings = model.encode(
    processed_text,
    convert_to_numpy=True,
    show_progress_bar=True
)

embeddings = embeddings.astype("float32")

faiss.normalize_L2(embeddings)


# ------------------------
# CLUSTERING
# ------------------------

def cluster_errors(embeddings, threshold=0.85):

    clusters = []
    visited = set()

    for i in range(len(embeddings)):

        if i in visited:
            continue

        cluster = [i]
        visited.add(i)

        for j in range(i + 1, len(embeddings)):

            sim = np.dot(embeddings[i], embeddings[j])

            if sim > threshold:

                cluster.append(j)
                visited.add(j)

        clusters.append(cluster)

    return clusters


clusters = cluster_errors(embeddings)

print("Total clusters:", len(clusters))


# ------------------------
# COMPUTE CLUSTER CENTROIDS
# ------------------------

cluster_centroids = []

for cluster in clusters:

    cluster_vecs = embeddings[cluster]

    centroid = np.mean(cluster_vecs, axis=0)

    cluster_centroids.append(centroid)


cluster_centroids = np.array(cluster_centroids).astype("float32")

faiss.normalize_L2(cluster_centroids)


# ------------------------
# FAISS INDEX ON CLUSTERS
# ------------------------

dimension = cluster_centroids.shape[1]

cluster_index = faiss.IndexFlatIP(dimension)

cluster_index.add(cluster_centroids)


# ------------------------
# SEARCH FUNCTION
# ------------------------

def search_error(query, testcase):

    template = extract_template(query)

    pattern = extract_pattern(template)

    combined = testcase + " " + template + " " + pattern

    query_embedding = model.encode([combined]).astype("float32")

    faiss.normalize_L2(query_embedding)

    # search clusters first
    k = 3

    distances, indices = cluster_index.search(query_embedding, k)

    best_cluster_id = indices[0][0]

    best_cluster = clusters[best_cluster_id]

    results = []

    for idx in best_cluster:

        sim = np.dot(query_embedding[0], embeddings[idx])

        if sim > 0.75:

            results.append({
                "row_number": row_ids[idx],
                "error": errors[idx],
                "similarity": float(sim)
            })

    return {
        "cluster_id": int(best_cluster_id),
        "frequency": len(best_cluster),
        "results": results
    }


# ------------------------
# TEST QUERY
# ------------------------

query = "hgfdfvc;xijhgvbhjk Error: Device 999 timeout after 1200 ms  fdfvc;xijhgvbhjk 45678909876t"

output = search_error(query, "pcie_reset_test")


print("\nCluster ID:", output["cluster_id"])

print("Error frequency:", output["frequency"])

print("\nSimilar Errors:")

for r in output["results"]:
    print(r)