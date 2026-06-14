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

Returns a JSON object with three fields:

```json
{
  "cuisine_type": "Mexican",
  "confidence_score": 95,
  "reasoning": "Quesadillas are a staple in Mexican cuisine..."
}
```

## How It Works

1. Loads the recipe dataset indexed by dish name
2. If **Look it up in the recipe dataset** is enabled, fetches ingredients and description for the dish
3. If **Use retrieval-augmented few-shot** is enabled, embeds the query and fetches 3 similar recipes from ChromaDB as few-shot examples
4. Builds a prompt with available context and few-shot examples, then sends it to `gemma2:2b`
5. Returns structured JSON with cuisine type, confidence score, and reasoning
6. Falls back to dish name only if the dish is not found in the dataset
7. Remaps any out-of-scope cuisine labels to `Other`

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

## Project Structure

```
src/cuisine_classifier/
├── __init__.py       # Package declaration
├── __main__.py       # Classifier logic and CLI entry point
├── config.yaml       # Model settings and cuisine types
└── README.md

retrieval/
├── vector_store.py   # ChromaDB wrapper (add + query)
├── populate.py       # Script to embed recipes into ChromaDB (200 at a time)
└── chroma_db/        # Persistent vector store (auto-created on first run)

app.py                # Streamlit UI (project root)
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
