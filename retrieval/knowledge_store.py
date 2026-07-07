"""
Cuisine knowledge card store for Week 6 RAG.

Embeds 6 cuisine knowledge cards (from data/cuisine_knowledge.md) into
retrieval/chroma_db_mpnet/ under collection 'cuisine_knowledge_mpnet',
using the fine-tuned mpnet model.  Isolated from the recipe collection
'cuisine_recipes_mpnet' — same client path, different collection name.

Usage:
    # Populate once (skips if already done):
    uv run python retrieval/knowledge_store.py

    # Query at runtime:
    from retrieval.knowledge_store import retrieve_cuisine_cards
    cards = retrieve_cuisine_cards("Chicken Teriyaki Tacos", "teriyaki soy sesame cabbage tortilla")
"""

import re
import os
import sys
import chromadb
from pathlib import Path

COLLECTION_NAME = "cuisine_knowledge_mpnet"
CHROMA_PATH = str(Path(__file__).parent / "chroma_db_mpnet")
MPNET_MODEL_PATH = str(Path(__file__).parent.parent / "models/ft-domain-embedding/cuisine_mpnet_ft")
KNOWLEDGE_MD = str(Path(__file__).parent.parent / "data/cuisine_knowledge.md")

_model_cache = None
_collection_cache = None


def _get_model():
    global _model_cache
    if _model_cache is None:
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer(MPNET_MODEL_PATH)
    return _model_cache


def get_knowledge_collection() -> chromadb.Collection:
    global _collection_cache
    if _collection_cache is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection_cache = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection_cache


def _parse_cuisine_knowledge(md_path: str) -> list:
    """Parse cuisine_knowledge.md into a list of card dicts."""
    text = Path(md_path).read_text(encoding="utf-8")
    cards = []
    blocks = re.split(r"^## ", text, flags=re.MULTILINE)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        cuisine = lines[0].strip()
        body = "\n".join(lines[1:]).strip()

        def _extract(text, start_marker, end_markers):
            pos = text.find(start_marker)
            if pos == -1:
                return ""
            content_start = pos + len(start_marker)
            end = len(text)
            for em in end_markers:
                p = text.find(em, content_start)
                if p != -1 and p < end:
                    end = p
            return text[content_start:end].strip()

        # Embed text = Core ingredients + Signals (everything before Format rules)
        fmt_pos = body.find("Format rules:")
        embed_text = body[:fmt_pos].strip() if fmt_pos != -1 else body

        cards.append({
            "cuisine": cuisine,
            "embed_text": embed_text,
            "format_rules": _extract(body, "Format rules:", ["Confused with:", "Trap:"]),
            "confused_with": _extract(body, "Confused with:", ["Trap:"]),
            "trap": _extract(body, "Trap:", []),
        })
    return cards


def populate_knowledge_store() -> None:
    """Embed 6 cuisine knowledge cards into ChromaDB. Skips if already populated."""
    col = get_knowledge_collection()
    if col.count() >= 6:
        print(f"Knowledge store ready: {col.count()} cards (already populated).")
        return

    cards = _parse_cuisine_knowledge(KNOWLEDGE_MD)
    if not cards:
        print("ERROR: No cards parsed from cuisine_knowledge.md", file=sys.stderr)
        return

    model = _get_model()
    texts = [c["embed_text"] for c in cards]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    col.add(
        ids=[c["cuisine"] for c in cards],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "cuisine":       c["cuisine"],
            "format_rules":  c["format_rules"],
            "confused_with": c["confused_with"],
            "trap":          c["trap"],
        } for c in cards],
    )
    print(f"Knowledge store ready: {col.count()} cards.")


def retrieve_cuisine_cards(dish_name: str, ingredients: str, n_results: int = 2) -> list:
    """Query cuisine knowledge ChromaDB for the top n cards.

    Returns list of dicts with keys: cuisine, format_rules, confused_with, trap, distance.
    Returns [] if the collection is not yet populated.
    """
    col = get_knowledge_collection()
    if col.count() == 0:
        return []

    model = _get_model()
    query = f"{dish_name}. {ingredients}" if ingredients else dish_name
    embedding = model.encode([query], show_progress_bar=False).tolist()

    results = col.query(
        query_embeddings=embedding,
        n_results=min(n_results, col.count()),
        include=["metadatas", "distances"],
    )

    cards = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        cards.append({
            "cuisine":       meta.get("cuisine", ""),
            "format_rules":  meta.get("format_rules", ""),
            "confused_with": meta.get("confused_with", ""),
            "trap":          meta.get("trap", ""),
            "distance":      round(dist, 4),
        })
    return cards


if __name__ == "__main__":
    populate_knowledge_store()
