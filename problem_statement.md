# Problem Statement

## Problem Context

The provided recipe dataset contains over 13,000 recipes with dish names, ingredients, and descriptions, but no cuisine type labels. Knowing the cuisine type of a recipe is useful for filtering, recommendation, and organizing recipe collections. An LLM can infer cuisine type from the combination of dish name, ingredients, and description without needing explicit labels in the data.

Over eight weeks, the classifier was progressively enhanced: from a basic dish-name-only prompt to a full multimodal pipeline with fine-tuned embeddings, a LoRA adapter, retrieval-augmented generation, and optional recipe card image input.

## Problem Statement

Given a dish name (and optionally a recipe card image), classify it into one of six cuisine types: Italian, Chinese, Mexican, Indian, American, or Other. The system auto-fetches ingredients and description from the dataset when available, or extracts them from an uploaded recipe card image. If neither is available, classification falls back to dish name only.

## LLM Role

Classification — the LLM picks one label from six fixed cuisine categories based on dish name, ingredients, description, and dynamically retrieved context (few-shot examples and cuisine knowledge cards).

## Full Pipeline (Weeks 1–7)

1. **User input** (Week 1) — types a dish name; optionally uploads a recipe card image
2. **Vision extraction** (Week 7) — if image uploaded, `llava:7b` extracts `dish_name`, `ingredients`, and `description`; malformed output corrected by `gemma2:2b`; structural and semantic quality checks applied
3. **Context lookup** (Week 1) — if no image or extraction fails, ingredients and description are fetched from the dataset CSV
4. **Embedding fine-tuning** (Week 2) — fine-tuned `all-mpnet-base-v2` on 120 cuisine domain pairs improves retrieval separability (cosine gap: 0.558 post fine-tune vs baseline)
5. **Few-shot retrieval** (Week 3) — ChromaDB queries ingredients + description to retrieve 3 semantically similar labeled recipes as dynamic few-shot examples
6. **Prompt engineering + Dataset labeling** (Week 4) — system prompt with format override rules and chain-of-thought reasoning instructions; Pydantic `CuisineResponse` schema enforces structured JSON output; 20 static boundary few-shot examples injected into the user prompt covering known confusion cases; separately, ~6K recipes labeled for cuisine using Claude Sonnet 4.6 API (output: `data/All_Recipe_Cuisine_Labeled_v2.csv`, `llm_cuisine` column) — validated on a 50-sample expert-verified holdout set, iterated prompt design on boundary-case errors to reach >90% holdout accuracy before full-scale labeling; used prompt caching on the static system prompt and incremental checkpointing every 100 records to prevent data loss
7. **LoRA adapter** (Week 5) — fine-tuned `gemma2:2b` adapter (`cuisine-classifier:latest`) trained on ~200 examples to fix name-keyword bias; achieves 90.5% accuracy vs 71% base model
8. **Ambiguity signal + Cuisine knowledge RAG** (Week 6) — k=3 neighbour retrieval computes ambiguity level; top 2 cuisine knowledge cards (format rules, confused-with, traps) retrieved from ChromaDB
9. **Classification** — all retrieved context fed to the LLM → cuisine label + reasoning + confidence score

## Example Flows

**Flow 1 — Context-enhanced base model:**
- Input: `Vegetable Quesadillas` (found in dataset)
- Ingredients and description auto-fetched from CSV
- Static few-shot examples + prompt rules applied
- Output: `Mexican`, confidence 95%

**Flow 2 — LoRA adapter (hard case):**
- Input: `Navajo Tacos`
- Base model incorrectly predicts `Mexican` (name-keyword bias on "Tacos")
- Adapter correctly predicts `American` (fry bread dish = Native American)
- Adapter fixes name-keyword bias cases that prompt rules alone cannot resolve

**Flow 3 — Cuisine knowledge RAG:**
- Input: `Bourbon Chicken`
- Ambiguity signal: `mild` — neighbours split between Chinese and American
- RAG retrieves Chinese cuisine card: trap — "despite 'bourbon', this is American-Chinese restaurant food"
- Output: `Chinese` — trap overrides the name signal

**Flow 4 — Image upload path:**
- Input: dish name `Chicken Teriyaki Tacos` + uploaded recipe card image
- Vision model extracts ingredients and description from the card
- Quality checks pass → extracted context fed into RAG pipeline
- Output: cuisine label + reasoning + confidence

**Flow 5 — Fallback (dish not in dataset, no image):**
- Input: `Butter Chicken`
- No context available — LLM classifies from dish name only
- Output: `Indian`, confidence 95%

## Success Criteria

- LLM returns valid JSON with `Cuisine`, `Reasoning`, and `Confidence_score` fields
- Cuisine value is one of the six defined categories
- **Base model accuracy (prompt-only):** 71% (30/42) on eval set
- **Context-enhanced accuracy:** 73.8% (31/42) on eval set (Macro F1: 73.6%)
- **Few-shot retrieval accuracy (mpnet):** 78.6% (33/42) on eval set (Macro F1: 79.5%, +4.8pp over context-enhanced)
- **Adapter accuracy:** 90.5% (38/42) on eval set
- **Cuisine knowledge RAG accuracy:** 90.5% (38/42) on eval set (Macro F1: 91.4%, +16.7pp over context-enhanced)
- Image upload path produces equivalent accuracy to CSV context enhancement for the same dish
- Quality check rejects non-recipe images before they reach the classification pipeline
