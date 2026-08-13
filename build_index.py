"""Rebuild disease_index.faiss from disease_database.csv.

app.py queries the index with a normalized query vector via
faiss.normalize_L2() and an inner-product search, so this script builds
a matching IndexFlatIP over normalized embeddings from the same
sentence-transformers model app.py loads at query time.

Run this after editing disease_database.csv, or any time
disease_index.faiss needs to be regenerated from scratch:

    python build_index.py
"""

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

CSV_PATH = "disease_database.csv"
INDEX_PATH = "disease_index.faiss"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    df = pd.read_csv(CSV_PATH)
    if "symptoms" not in df.columns:
        raise ValueError(f"{CSV_PATH} must have a 'symptoms' column")

    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(df)} rows...")
    embeddings = model.encode(
        df["symptoms"].astype(str).tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    print(f"Wrote {index.ntotal} vectors ({embeddings.shape[1]}-dim) to {INDEX_PATH}")


if __name__ == "__main__":
    main()
