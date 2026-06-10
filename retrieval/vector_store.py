"""
ChromaDB vector store for cuisine recipe retrieval.

Usage:
    vs = VectorStore()
    vs.add(texts, labels, metadata_list)
    results = vs.query("zucchini cheese tortilla", k=3)
"""

import ollama
import chromadb
from datetime import datetime, timezone
from pathlib import Path

EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "cuisine_recipes"
CHROMA_PATH = str(Path(__file__).parent / "chroma_db")
BATCH_SIZE = 100


class VectorStore:
    def __init__(self, path=CHROMA_PATH):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self):
        return self.collection.count()

    def add(self, ids, texts, labels, metadata_list):
        existing_ids = set(self.collection.get()["ids"])
        rows = [
            (id_, text, label, meta)
            for id_, text, label, meta in zip(ids, texts, labels, metadata_list)
            if id_ not in existing_ids
        ]

        if not rows:
            print("All documents already stored — nothing to add.")
            return

        print(f"Adding {len(rows)} new documents (skipping {len(ids) - len(rows)} already stored)...")
        timestamp = datetime.now(timezone.utc).isoformat()

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            batch_ids   = [r[0] for r in batch]
            batch_texts = [r[1] for r in batch]
            batch_labels = [r[2] for r in batch]
            batch_meta  = [r[3] for r in batch]

            embeddings = ollama.embed(model=EMBED_MODEL, input=batch_texts)["embeddings"]

            for meta, label in zip(batch_meta, batch_labels):
                meta["cuisine_type"] = label
                meta["embedding_model"] = EMBED_MODEL
                meta["embedding_timestamp"] = timestamp

            self.collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )
            print(f"  {min(i + BATCH_SIZE, len(rows))}/{len(rows)} done")

        print(f"Done. Collection now has {self.collection.count()} documents.")

    def query(self, text, k=3):
        embedding = ollama.embed(model=EMBED_MODEL, input=[text])["embeddings"]
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "document": doc,
                "dish_name": meta.get("dish_name", ""),
                "cuisine_type": meta.get("cuisine_type", ""),
                "ingredient_count": meta.get("ingredient_count", ""),
                "distance": round(dist, 4),
            })
        return output
