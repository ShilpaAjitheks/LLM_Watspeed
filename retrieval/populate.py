"""
Populate ChromaDB with labeled recipes from the dataset, 200 at a time.

Uses All_Recipe_Cuisine_Labeled_v2.csv which contains llm_cuisine labels.

Stratified sampling: caps each cuisine class at MAX_PER_CLASS to balance retrieval
quality across all six cuisines (default 400). Classes already over the cap are
skipped; classes with fewer total recipes use all available.

Run each time you want to add the next 200 recipes:
    uv run python retrieval/populate.py                  # nomic-embed-text (default)
    uv run python retrieval/populate.py --model mpnet    # fine-tuned mpnet

Skips recipes already stored — safe to re-run.
"""

import sys
import argparse
import pandas as pd
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieval.vector_store import VectorStore, NOMIC, MPNET

DATASET_PATH = Path(__file__).parent.parent / "data" / "All_Recipe_Cuisine_Labeled_v2.csv"
BATCH_LIMIT = 200
MAX_PER_CLASS = 88


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[NOMIC, MPNET], default=NOMIC,
                        help="Embedding model to use (default: nomic-embed-text)")
    args = parser.parse_args()

    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["Description", "Ingredients", "llm_cuisine"]).reset_index(drop=True)
    print(f"Loaded {len(df)} labeled recipes from {DATASET_PATH.name}.")

    vs = VectorStore(model=args.model)
    existing_ids = set(vs.collection.get()["ids"])
    print(f"Model: {args.model}")
    print(f"Collection currently has {len(existing_ids)} documents.")

    # Count how many per class are already stored
    stored_meta = vs.collection.get(include=["metadatas"])["metadatas"]
    class_counts = Counter(m.get("cuisine_type", "") for m in stored_meta)
    print("Current per-class counts:")
    for label, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<12} {count:>5} / {MAX_PER_CLASS}")

    # Filter: skip already stored AND skip classes already at cap
    new_rows = [
        (i, row) for i, row in df.iterrows()
        if str(i) not in existing_ids
        and class_counts.get(str(row["llm_cuisine"]), 0) < MAX_PER_CLASS
    ]

    if not new_rows:
        print("All classes are at their cap or no new recipes available.")
        return

    next_batch = new_rows[:BATCH_LIMIT]
    remaining_after = len(new_rows) - len(next_batch)
    batch_class_counts = Counter(str(row["llm_cuisine"]) for _, row in next_batch)
    print(f"\nEmbedding next {len(next_batch)} recipes ({remaining_after} remaining after this run)...")
    print("This batch breakdown:", dict(batch_class_counts))

    ids    = [str(i) for i, _ in next_batch]
    texts  = [str(row["Name"]) + " " + str(row["Ingredients"]).replace("|", ", ") + " " + str(row["Description"]) for _, row in next_batch]
    labels = [str(row["llm_cuisine"]) for _, row in next_batch]
    metas  = [
        {
            "dish_name":        str(row["Name"]),
            "ingredient_count": len(str(row["Ingredients"]).split("|")),
        }
        for _, row in next_batch
    ]

    vs.add(ids, texts, labels, metas)
    print(f"Done. Run again to embed the next {BATCH_LIMIT}.")


if __name__ == "__main__":
    main()
