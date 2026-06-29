# Cuisine Classifier

A modular recipe cuisine classifier built with `gemma2:2b` via Ollama. Given a dish name, it auto-fetches ingredients and description from the recipe dataset and classifies the dish into one of six cuisine types.

## Supported Cuisine Types

Italian, Chinese, Mexican, Indian, American, Other

> **Other** is used for dishes that don't clearly belong to any of the five main categories.

## Usage

### CLI

```bash
# Classify a dish (with dataset context)
uv run python -m cuisine_classifier "Vegetable Quesadillas"

# Classify a dish not in the dataset (falls back to dish name only)
uv run python -m cuisine_classifier "Butter Chicken"

# Force dish name only (skip dataset lookup)
uv run python -m cuisine_classifier "Pasta" --no-context

# Use a custom dataset
uv run python -m cuisine_classifier "Spaghetti" --dataset path/to/recipes.csv
```

### Streamlit App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), enter a dish name, and click **Classify** to see the cuisine type, confidence score, and reasoning.

Two optional checkboxes are available:
- **Look it up in the recipe dataset** — fetches ingredients and description for richer context
- **Use retrieval-augmented few-shot** — retrieves 3 similar recipes from ChromaDB and uses them as few-shot examples in the prompt. Requires the vector store to be populated first (see below).

## Output

Returns a structured response with four fields (enforced via Pydantic `CuisineResponse`):

```json
{
  "recipe_name": "Vegetable Quesadillas",
  "thoughts": [
    "Quesadillas are a staple Mexican dish",
    "Tortillas and cheese filling confirm Mexican origin"
  ],
  "confidence_score": 95,
  "cuisine": "Mexican"
}
```

- `thoughts` — chain-of-thought reasoning steps the model worked through before classifying
- `confidence_score` — 0–100; note: small models can be correct but uncertain (score near 0) on boundary cases
- `cuisine` — one of the six fixed labels; remapped to `Other` if the model returns anything outside the list

## How It Works

1. Loads the recipe dataset indexed by dish name
2. Looks up ingredients and description for the dish (falls back to dish name only if not found)
3. Builds a prompt using `build_prompt()`:
   - Always prepends **16 static boundary few-shot examples** (`_STATIC_FEW_SHOT`) covering known confusion cases (e.g. "Italian Wedding Cookies → American", "Yaki Mandu → Other")
   - If ChromaDB retrieval is enabled, appends 3 dynamically retrieved similar recipes (queried by ingredients + description, or dish name as fallback)
4. Sends the prompt via `ollama.chat()` with a structured system prompt containing:
   - Culinary expert persona
   - Chain-of-thought instruction ("Think step by step in 2-3 short lines")
   - Origin-priority rule (classify by culinary origin, not ingredient adaptations)
   - Other-category guidance (explicit list of cuisines that belong in Other)
5. Parses the response into a `CuisineResponse` Pydantic model — enforces schema at the model level
6. Remaps any out-of-scope cuisine labels to `Other`

## Semantic Retrieval (Week 3)

The classifier supports retrieval-augmented few-shot prompting using ChromaDB. Two embedding models are supported, each with its own separate ChromaDB store:

| Model | Backend | ChromaDB path |
|---|---|---|
| `nomic-embed-text` | Ollama | `retrieval/chroma_db/` |
| `mpnet` | fine-tuned all-mpnet-base-v2 (local CPU) | `retrieval/chroma_db_mpnet/` |

The model is selected via a radio button in the Streamlit UI when retrieval is enabled.

### Populate the vector store

Run this to embed the next 200 recipes. Re-run each time you want to add more:

```bash
# nomic-embed-text (default)
uv run python retrieval/populate.py

# fine-tuned mpnet
uv run python retrieval/populate.py --model mpnet
```

Each run skips recipes already stored. Check current count:

```bash
uv run python -c "import sys; sys.path.insert(0, '.'); from retrieval.vector_store import VectorStore; vs = VectorStore(); print('nomic count:', vs.count())"
uv run python -c "import sys; sys.path.insert(0, '.'); from retrieval.vector_store import VectorStore; vs = VectorStore(model='mpnet'); print('mpnet count:', vs.count())"
```

### How retrieval works

- Each recipe is stored as `ingredients + description` text, embedded with the selected model
- At query time, the dish's ingredients + description is embedded and the 3 nearest recipes are retrieved
- Those 3 recipes are prepended to the prompt as few-shot examples
- The query dish itself is excluded from retrieved results to avoid self-referencing

### Metadata stored per recipe

| Field | Source |
|---|---|
| `dish_name` | `Name` column |
| `ingredient_count` | derived from `Ingredients` (count of `\|` separators) |
| `cuisine_type` | empty until ground truth labels are generated via SOTA model |
| `embedding_model` | `nomic-embed-text` or `mpnet` |
| `embedding_timestamp` | ISO timestamp at embed time |

## Evaluation

Run the ablation study against the 30-dish golden eval set (`data/prompt_eval_cuisine.csv`):

```bash
# Two-leg ablation: baseline (dish name only) vs. context-enhanced (dataset lookup)
uv run python -m cuisine_classifier.evaluate --verbose

# Skip baseline leg — run context-enhanced only (faster)
uv run python -m cuisine_classifier.evaluate --verbose --skip-baseline

# Three-leg ablation: add retrieval-augmented as third leg
uv run python -m cuisine_classifier.evaluate --verbose --retrieval --embed-model all-mpnet-base-v2
```

The script runs each dish one by one and prints a live counter (`[No Context] 1/30 — dish name`). At the end it shows per-leg accuracy, failures, per-category breakdown, and improvement comparison.

| Leg | Mode | Flag |
|---|---|---|
| 1 | Baseline — dish name only | always runs |
| 2 | Context-enhanced — dataset lookup | always runs |
| 3 | Retrieval-augmented — context + ChromaDB | `--retrieval --embed-model <model>` |

**Week 4 benchmark:** context-enhanced accuracy = 83.3% (25/30) on the 30-dish eval set.

### Changing or expanding the eval set

Edit `data/prompt_eval_cuisine.csv` directly — add or remove rows as needed. The format is two columns:

```csv
Name,expected
Your Dish Name,Italian
Another Dish,Chinese
```

The eval script also accepts `.xlsx` files and recognises `expert_label` as an alias for the `expected` column, so spreadsheets exported from a labelling tool work without renaming columns.

To use a different file entirely:

```bash
uv run python -m cuisine_classifier.evaluate --eval-set data/my_custom_eval.csv --verbose
uv run python -m cuisine_classifier.evaluate --eval-set data/validation_sample.xlsx --verbose
```

Any number of dishes is supported. Each dish runs 2 model calls (one per leg), so time scales linearly — 30 dishes ≈ 5–10 min, 60 dishes ≈ 10–20 min. The live counter keeps you informed throughout.

> **Week 5 tip:** consider expanding to 50–60 dishes before running the LoRA fine-tuning comparison — more coverage gives a more reliable accuracy gap between prompt-only and adapter-based results.

## Embedding Fine-Tuning (Week 3)

`embedder.py` fine-tunes `all-mpnet-base-v2` on cuisine domain pairs so the embedding space better separates dishes by cuisine. The fine-tuned model is then available as the `mpnet` option in the retrieval pipeline.

> **Hardware:** designed to run in Google Colab with a GPU. CPU execution works but is significantly slower.

### Domain pairs

`domain_pairs.py` provides 120 labeled training pairs — 20 per cuisine (6 cuisines), split evenly between `close` (dish belongs to that cuisine) and `far` (dish is from a different cuisine). 60 pairs are backed by real AllRecipes dataset entries; 60 use general dish terms.

### Running the fine-tuner

```bash
uv run python src/cuisine_classifier/embedder.py
```

This will:
1. Load all 120 domain pairs from `domain_pairs.py`
2. Measure the baseline cosine similarity gap between close/far pairs using the base `all-mpnet-base-v2` model
3. Fine-tune for 5 epochs with CoSENTLoss (train/eval split 85/15)
4. Compare fine-tuned gap vs baseline gap and print per-pair deltas
5. Save the fine-tuned model to `models/ft-domain-embedding/cuisine_mpnet_ft`
6. Save evaluation results to `data/embedding_eval.csv`

A gap ≥ 0.2 indicates good cuisine signal; < 0.1 suggests the base model or pairs need revision.

### Using the fine-tuned model

Once saved, the model can be used in the retrieval pipeline and evaluation:

```bash
# Populate ChromaDB with the fine-tuned model
uv run python retrieval/populate.py --model mpnet

# Run evaluation with retrieval using the fine-tuned model
uv run python -m cuisine_classifier.evaluate --verbose --retrieval --embed-model all-mpnet-base-v2
```

## Project Structure

```
src/cuisine_classifier/
├── __init__.py         # Package declaration
├── __main__.py         # Classifier logic and CLI entry point
├── evaluate.py         # Evaluation script — ablation study on golden eval set
├── embedder.py         # Fine-tuning script for all-mpnet-base-v2 on domain pairs
├── domain_pairs.py     # 120 labeled close/far pairs for embedding fine-tuning
├── config.yaml         # Model settings and cuisine types
└── README.md

retrieval/
├── vector_store.py     # ChromaDB wrapper (add + query)
├── populate.py         # Script to embed recipes into ChromaDB (200 at a time)
└── chroma_db/          # Persistent vector store (auto-created on first run)

models/
└── ft-domain-embedding/
    └── cuisine_mpnet_ft/   # Fine-tuned all-mpnet-base-v2 (output of embedder.py)

app.py                  # Streamlit UI (project root)
```

## Configuration

Edit `config.yaml` to change the model, dataset path, or cuisine types:

```yaml
model:
  name: "gemma2:2b"
  temperature: 0.0

dataset:
  path: "data/All_Recipe_Web_Scraping_Dataset_Labeled.csv"

cuisine_types:
  - Italian
  - Chinese
  - Mexican
  - Indian
  - American
  - Other
```
