"""
Populate ChromaDB with recipes from the dataset, 200 at a time.

Run each time you want to add the next 200 recipes:
    uv run python retrieval/populate.py

Skips recipes already stored — safe to re-run.
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieval.vector_store import VectorStore

DATASET_PATH = Path(__file__).parent.parent / "data" / "All_Recipe_Web_Scraping_Dataset_Labeled.csv"
BATCH_LIMIT = 200


def main():
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["Description", "Ingredients"]).reset_index(drop=True)

    vs = VectorStore()
    existing_ids = set(vs.collection.get()["ids"])
    print(f"Collection currently has {len(existing_ids)} documents.")

    new_rows = [(i, row) for i, row in df.iterrows() if str(i) not in existing_ids]

    if not new_rows:
        print("All recipes already stored.")
        return

    next_batch = new_rows[:BATCH_LIMIT]
    print(f"Embedding next {len(next_batch)} recipes ({len(new_rows) - len(next_batch)} remaining after this run)...")

    ids    = [str(i) for i, _ in next_batch]
    texts  = [str(row["Ingredients"]).replace("|", ", ") + " " + str(row["Description"]) for _, row in next_batch]
    labels = [""] * len(next_batch)
    metas  = [
        {
            "dish_name": str(row["Name"]),
            "ingredient_count": len(str(row["Ingredients"]).split("|")),
        }
        for _, row in next_batch
    ]

    vs.add(ids, texts, labels, metas)
    print(f"Done. Run again to embed the next {BATCH_LIMIT}.")


if __name__ == "__main__":
    main()
