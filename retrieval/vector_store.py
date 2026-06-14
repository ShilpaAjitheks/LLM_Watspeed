"""
ChromaDB vector store for cuisine recipe retrieval.

Supports two embedding backends:
  - "nomic-embed-text" (Ollama) — stored in retrieval/chroma_db/
  - "mpnet"  (fine-tuned all-mpnet-base-v2) — stored in retrieval/chroma_db_mpnet/

Usage:
    vs = VectorStore()                      # nomic (default)
    vs = VectorStore(model="mpnet")         # fine-tuned mpnet
    vs.add(ids, texts, labels, metadata_list)
    results = vs.query("zucchini cheese tortilla", k=3)
"""

import ollama
import chromadb
from datetime import datetime, timezone
from pathlib import Path

NOMIC = "nomic-embed-text"
MPNET = "mpnet"
MPNET_MODEL_PATH = str(Path(__file__).parent.parent / "models/ft-domain-embedding/cuisine_mpnet_ft")

BACKENDS = {
    NOMIC: {
        "chroma_path": str(Path(__file__).parent / "chroma_db"),
        "collection": "cuisine_recipes",
    },
    MPNET: {
        "chroma_path": str(Path(__file__).parent / "chroma_db_mpnet"),
        "collection": "cuisine_recipes_mpnet",
    },
}

BATCH_SIZE = 100


class VectorStore:
    def __init__(self, model=NOMIC):
        if model not in BACKENDS:
            raise ValueError(f"model must be one of {list(BACKENDS.keys())}")

        self.model = model
        backend = BACKENDS[model]
        self.client = chromadb.PersistentClient(path=backend["chroma_path"])
        self.collection = self.client.get_or_create_collection(
            name=backend["collection"],
            metadata={"hnsw:space": "cosine"},
        )

        self._st_model = None
        if model == MPNET:
            import os
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(MPNET_MODEL_PATH)

    def _embed(self, texts):
        if self.model == NOMIC:
            return ollama.embed(model=NOMIC, input=texts)["embeddings"]
        else:
            return self._st_model.encode(texts, show_progress_bar=False).tolist()

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
            batch_ids    = [r[0] for r in batch]
            batch_texts  = [r[1] for r in batch]
            batch_labels = [r[2] for r in batch]
            batch_meta   = [r[3] for r in batch]

            embeddings = self._embed(batch_texts)

            for meta, label in zip(batch_meta, batch_labels):
                meta["cuisine_type"]        = label
                meta["embedding_model"]     = self.model
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
        embedding = self._embed([text])
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
                "document":         doc,
                "dish_name":        meta.get("dish_name", ""),
                "cuisine_type":     meta.get("cuisine_type", ""),
                "ingredient_count": meta.get("ingredient_count", ""),
                "distance":         round(dist, 4),
            })
        return output
