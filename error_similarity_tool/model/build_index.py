import pandas as pd
import re
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

df = pd.read_excel("data/error_dataset_with_testcases_2500.xlsx")

errors = df["error_message"].tolist()
row_ids = df["row_number"].tolist()
testcases = df["testcase"].tolist()


def extract_template(msg):

    msg = msg.lower()

    msg = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', msg)
    msg = re.sub(r'\d+', '<NUM>', msg)

    return msg


processed = []

for err, tc in zip(errors, testcases):

    template = extract_template(err)

    processed.append(tc + " " + template)


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(processed, convert_to_numpy=True)

embeddings = embeddings.astype("float32")

faiss.normalize_L2(embeddings)

dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)

index.add(embeddings)

faiss.write_index(index, "model/faiss_index.bin")

tokenized = [text.split() for text in processed]

bm25 = BM25Okapi(tokenized)

pickle.dump(bm25, open("model/bm25.pkl", "wb"))

pickle.dump(errors, open("model/errors.pkl", "wb"))
pickle.dump(row_ids, open("model/rows.pkl", "wb"))
pickle.dump(testcases, open("model/testcases.pkl", "wb"))

print("Index built successfully")