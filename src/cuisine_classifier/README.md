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

# Context-enhanced + few-shot retrieval (fine-tuned mpnet, Week 3)
uv run python -m cuisine_classifier "Bourbon Chicken" --retrieval

# Context-enhanced + few-shot retrieval using nomic embeddings
uv run python -m cuisine_classifier "Bourbon Chicken" --retrieval --embed-model nomic-embed-text

# Context-enhanced + retrieval + cuisine knowledge RAG (all layers)
uv run python -m cuisine_classifier "Bourbon Chicken" --retrieval --rag

# Use Week 6 cuisine knowledge RAG only
uv run python -m cuisine_classifier "Bourbon Chicken" --rag

# Classify from a recipe card image (Week 7 — extracts ingredients and description via llava:7b)
uv run python -m cuisine_classifier "Chicken Teriyaki Tacos" --image path/to/recipe_card.png
```

### Streamlit App

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), enter a dish name, and click **Classify** to see the cuisine type, confidence score, and reasoning.

Controls available:
- **Model** radio — three options:
  - **Base model** (`gemma2:2b`) — prompt-only classifier
  - **Adapter (Option B)** (`cuisine-classifier:latest`) — Week 5 LoRA adapter; shows a warning if not registered in Ollama
  - **Compare both** — runs both models on the same input and shows results side by side in two columns
- **Look it up in the recipe dataset** — fetches ingredients and description for richer context
- **Use retrieval-augmented few-shot** — retrieves 3 similar recipes from ChromaDB and uses them as few-shot examples in the prompt. Requires the vector store to be populated first (see below).
- **Use cuisine knowledge RAG** — Week 6 two-layer RAG: Layer 1 computes an ambiguity signal (k=3 neighbour retrieval), Layer 2 retrieves the top 2 cuisine knowledge cards from ChromaDB. Requires the mpnet vector store and knowledge store to be populated first. Only applies to the base model (the adapter uses its own trained format).
- **Upload a recipe card (optional)** — Week 7 image upload: extracts ingredients and description from a recipe card photo using `llava:7b` (vision model) with `gemma2:2b` as a corrector for malformed output. Runs structural completeness and semantic plausibility checks before feeding extracted context into the pipeline. Falls back to dataset context if image is rejected.

## Output

Returns a structured response with four fields (enforced via Pydantic `CuisineResponse`):

```json
{
  "Recipe_name": "Vegetable Quesadillas",
  "Reasoning": [
    "Quesadillas are a staple Mexican dish",
    "Tortillas and cheese filling confirm Mexican origin"
  ],
  "Confidence_score": 95,
  "Cuisine": "Mexican"
}
```

- `Reasoning` — chain-of-thought reasoning steps the model worked through before classifying
- `Confidence_score` — 0–100; note: small models can be correct but uncertain (score near 0) on boundary cases
- `Cuisine` — one of the six fixed labels; remapped to `Other` if the model returns anything outside the list

## How It Works

1. Loads the recipe dataset indexed by dish name
2. Looks up ingredients and description for the dish (falls back to dish name only if not found)
3. Builds a prompt using `build_prompt()`:
   - Always prepends **20 static boundary few-shot examples** (`_STATIC_FEW_SHOT`) covering known confusion cases (e.g. "Italian Wedding Cookies → American", "Bourbon Chicken → Chinese", "Twinkie® Tiramisu → Italian")
   - If ChromaDB retrieval is enabled, appends 3 dynamically retrieved similar recipes (queried by ingredients + description, or dish name as fallback)
4. Sends the prompt to Ollama — two paths depending on the model:
   - **Base model** (`gemma2:2b`): `ollama.chat()` with separate system/user roles + JSON schema enforcement via `format=CuisineResponse.model_json_schema()`
   - **Adapter** (`cuisine-classifier:latest`): `ollama.generate()` with the Gemma2 chat template applied manually (`<bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`) and free-form JSON parsing — matches the exact inference format used during Colab training
5. Parses the response:
   - Base model: `CuisineResponse.model_validate_json()` (strict Pydantic)
   - Adapter: `_parse_free_form()` — regex-based extraction with `_AMERICAN_ALIASES` remapping
6. Remaps any out-of-scope cuisine labels to `Other`

## Semantic Retrieval (Week 3)

The classifier supports retrieval-augmented few-shot prompting using ChromaDB. Two embedding models are supported, each with its own separate ChromaDB store:

| Model | Backend | ChromaDB path |
|---|---|---|
| `nomic-embed-text` | Ollama | `retrieval/chroma_db/` |
| `mpnet` | fine-tuned all-mpnet-base-v2 (local CPU) | `retrieval/chroma_db_mpnet/` |

The model is selected via a radio button in the Streamlit UI when retrieval is enabled.

### Populate the vector store

Run this to embed the next 200 recipes. Re-run each time you want to add more. Already-stored recipes are skipped, so it's safe to re-run:

> **To embed the full dataset in one pass:** set `BATCH_LIMIT = 6781` in `retrieval/populate.py` before running — the dataset has ~6,781 usable recipes in total.

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

Run the ablation study against an eval set:

```bash
# Two-leg ablation: baseline (dish name only) vs. context-enhanced (dataset lookup)
uv run python -m cuisine_classifier.evaluate --verbose

# Skip baseline leg — run context-enhanced only (faster)
uv run python -m cuisine_classifier.evaluate --verbose --skip-baseline

# Three-leg ablation: add retrieval-augmented as third leg
uv run python -m cuisine_classifier.evaluate --verbose --retrieval --embed-model mpnet

# Use the Week 5 edge-case eval set
uv run python -m cuisine_classifier.evaluate --eval-set data/prompt_eval_cuisine_fin.csv --skip-baseline --verbose

# Week 6 RAG ablation leg — context-enhanced + cuisine knowledge RAG
uv run python -m cuisine_classifier.evaluate --eval-set data/prompt_eval_cuisine_fin.csv --skip-baseline --rag --verbose

# Compare base model vs adapter on the same eval set
uv run python -m cuisine_classifier.evaluate --eval-set data/prompt_eval_cuisine_fin.csv --skip-baseline --verbose
uv run python -m cuisine_classifier.evaluate --eval-set data/prompt_eval_cuisine_fin.csv --model cuisine-classifier:latest --skip-baseline --verbose
```

The script runs each dish one by one and prints a live counter. At the end it shows per-leg accuracy, failures, per-category breakdown, and improvement comparison.

| Flag | Purpose |
|---|---|
| `--eval-set` | Path to eval CSV or XLSX (`Name`, `expected` columns) |
| `--model` | Ollama model name to use (overrides config default `gemma2:2b`) |
| `--skip-baseline` | Skip the dish-name-only leg, run context-enhanced only |
| `--verbose` | Show every prediction, not just failures |
| `--retrieval --embed-model <mpnet\|nomic-embed-text>` | Add few-shot retrieval (Week 3) as a third leg — queries ChromaDB for 3 similar recipes and injects them as dynamic few-shot examples. `--embed-model` is required when using `--retrieval` |
| `--rag` | Add Week 6 cuisine knowledge RAG as an ablation leg — retrieves cuisine knowledge cards (format rules, traps) from ChromaDB; becomes Leg 4 when combined with `--retrieval` |

**Leg numbering:**

| Flags used | Leg 2 | Leg 3 | Leg 4 |
|---|---|---|---|
| `--skip-baseline` | Context-Enhanced | — | — |
| `--skip-baseline --retrieval --embed-model mpnet` | Context-Enhanced | Few-shot Retrieval (Week 3) | — |
| `--skip-baseline --rag` | Context-Enhanced | Cuisine Knowledge RAG (Week 6) | — |
| `--skip-baseline --retrieval --embed-model mpnet --rag` | Context-Enhanced | Few-shot Retrieval (Week 3) | Cuisine Knowledge RAG (Week 6) |

> Leg 1 (dish name only) is included when `--skip-baseline` is omitted.

**Eval sets:**

| File | Description |
|---|---|
| `data/prompt_eval_cuisine.csv` | Original 30-dish golden eval set (Weeks 1–4) |
| `data/prompt_eval_cuisine_fin.csv` | Edge-case-heavy set used for Week 5 adapter comparison |

The eval script also accepts `.xlsx` files and recognises `expert_label` as an alias for the `expected` column.

## LoRA Adapter Fine-Tuning (Week 5)

A LoRA adapter was trained on ~200 cuisine classification examples to fix **name-keyword bias** — cases where the dish name (e.g. "taco", "lasagna") overrides ingredient signals and produces the wrong label.

**Training setup:**
- Base model: `unsloth/gemma-2-2b-it-bnb-4bit`
- LoRA: r=16, alpha=32, trained with Unsloth SFTTrainer on Google Colab GPU
- Exported as GGUF (q8_0) and registered in Ollama as `cuisine-classifier:latest`
- Training format matches inference exactly: system prompt + few-shot user prompt + `CuisineResponse` JSON

**Register the adapter (if not already done):**
```bash
ollama create cuisine-classifier -f Modelfile
```

**Verify it's registered:**
```bash
ollama list | grep cuisine
```

**Test on known hard cases:**
```bash
# Name-keyword bias: "taco" overrides fry-bread ingredients → base says Mexican, adapter says American
uv run python -m cuisine_classifier "Navajo Tacos"

# Regression check: phyllo pastry format overrides spice profile → both should say Other
uv run python -m cuisine_classifier "Easy Baklava"

# Regression check: sour cream / cream cheese markers → American, not Italian
uv run python -m cuisine_classifier "Grandma's Best Ever Sour Cream Lasagna"
```

**Success threshold:** adapter accuracy ≥ 74% on `prompt_eval_cuisine_fin.csv` with no regressions on Baklava (→ Other) or American-style lasagna (→ American).

**Actual results (2026-06-30, 42-dish eval set):**

| Model | Accuracy | Notes |
|---|---|---|
| `gemma2:2b` (base) | 71% (30/42) | prompt-only |
| `cuisine-classifier:latest` (Colab) | 93% (39/42) | measured in Colab training environment |
| `cuisine-classifier:latest` (local) | **90.5% (38/42)** | `ollama.generate()` + manual Gemma2 template |

Key fix: switching from `ollama.chat()` to `ollama.generate()` with the Gemma2 chat template applied manually closed the Colab vs local gap (76.2% → 90.5%). Remaining 4 failures (Beef Stroganoff, Potato Curry, Lamb Patties, Swedish Meatballs I) need additional training examples.

---

## Cuisine Knowledge RAG (Week 6)

A two-layer retrieval-augmented generation pipeline that replaces the static hardcoded system prompt rules with dynamically retrieved cuisine knowledge cards.

**Layer 1 — Ambiguity signal:** queries the mpnet recipe vector store with k=3 neighbours. Returns level (`confident` / `mild` / `high`), the candidate cuisines, and a note (e.g. "neighbours split across ['Mexican', 'Chinese']").

**Layer 2 — Cuisine knowledge cards:** independently queries a separate ChromaDB collection (`cuisine_knowledge_mpnet`) using dish name + ingredients as the query. Returns the top 2 cuisine cards, each containing Format rules, Confused-with, and Trap fields.

Both signals are combined into the prompt. The model is explicitly instructed: *"If a retrieved trap names this exact dish, follow it over all other signals."*

**Knowledge cards** are stored in `data/cuisine_knowledge.md` — 6 cards (one per cuisine) created offline. Each card has:
- **Body** (embedded): Core ingredients + Signals — used for similarity search
- **Metadata** (not embedded, passed to LLM): Format rules, Confused with, Trap

**Populate the knowledge store (run once):**
```bash
uv run python retrieval/knowledge_store.py
```

**Results on `prompt_eval_cuisine_fin.csv` (42 dishes):**

| Mode | Accuracy | Notes |
|---|---|---|
| Context-Enhanced | 73.8% (31/42) | dataset lookup only |
| Cuisine Knowledge RAG | **90.5% (38/42)** | +16.7pp over context-enhanced |

Remaining 3 failures: Stuffed Peppers, Hoisin-Glazed Salmon, Gluten-Free Biscotti — all cases where the Other knowledge card was ranked first in Layer 2 retrieval.

---

## Vision Extraction (Week 7)

Adds optional recipe card image upload as an alternative to CSV context enhancement. When an image is uploaded, `vision.py` runs a full extraction and quality check pipeline before feeding context into the Week 6 RAG pipeline.

**Extraction pipeline (`vision.py`):**
1. `llava:7b` reads the image and returns JSON with `dish_name`, `ingredients` (list), `description` (string)
2. Markdown fences stripped; if JSON parse fails, `gemma2:2b` corrector reformats the output
3. If correction also fails, returns `None` and surfaces raw output

**Quality checks (`check_extraction`):**
- Structural completeness: all three fields present with correct types (`ingredients` must be a list, `description` must be a string)
- Semantic plausibility: at least 3 ingredients, non-empty description, dish name similarity ≥ 0.75 (fuzzy match via `SequenceMatcher`) against typed dish name

**Note on model choice:** `gemma4:e4b` (used in the notebook on Colab/Linux) crashes on Windows with a stack buffer overrun in Ollama. `llava:7b` is used locally as a drop-in replacement with the same API. Switch back to `gemma4:e4b` by changing `VISION_MODEL` in `vision.py` when running on Linux/Colab.

---

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
├── __main__.py         # Classifier logic and CLI entry point; predict_cuisine() wires all
│                       #   pipeline layers: context lookup, retrieval, RAG, vision, adapter
├── evaluate.py         # Ablation study: baseline → context-enhanced → retrieval → RAG legs;
│                       #   reports accuracy and macro F1 per leg
├── embedder.py         # Fine-tunes all-mpnet-base-v2 on 120 domain pairs (CoSENTLoss, 5 epochs);
│                       #   saves to models/ft-domain-embedding/cuisine_mpnet_ft
├── domain_pairs.py     # 120 labeled close/far pairs (20 per cuisine) for embedding fine-tuning
├── config.yaml         # Model name, temperature, dataset path, cuisine types
├── vision.py           # Week 7: llava:7b extracts dish_name/ingredients/description from image;
│                       #   gemma2:2b corrects malformed JSON; structural + semantic quality checks
└── README.md

retrieval/
├── vector_store.py     # ChromaDB wrapper for the recipe few-shot collection; supports
│                       #   nomic-embed-text (Ollama) and mpnet (local fine-tuned) backends;
│                       #   used to retrieve 3 similar labeled recipes as dynamic few-shot examples
├── populate.py         # Embeds recipes from dataset into ChromaDB for few-shot retrieval;
│                       #   runs in batches of 200, skips already-stored recipes;
│                       #   supports --model nomic-embed-text (default) and --model mpnet
├── knowledge_store.py  # Week 6: embeds cuisine knowledge cards (format rules, confused-with,
│                       #   traps — one card per cuisine) into a separate ChromaDB collection
│                       #   (cuisine_knowledge_mpnet); retrieves top 2 cards per query for RAG
├── chroma_db/          # Persistent vector store — nomic-embed-text recipe embeddings (few-shot)
└── chroma_db_mpnet/    # Persistent vector store — mpnet embeddings; two separate collections:
│                       #   recipe few-shot collection + cuisine knowledge card collection

models/
├── ft-domain-embedding/
│   └── cuisine_mpnet_ft/   # Fine-tuned all-mpnet-base-v2 (output of embedder.py)
└── (GGUF adapter placed here or at project root)

Modelfile               # Ollama Modelfile pointing to the GGUF adapter (cuisine-classifier:latest)
cuisine_adapter_q4.gguf # LoRA adapter weights — q8_0 quantization (named q4 historically)
app.py                  # Streamlit UI — dish name input, image upload, model/retrieval/RAG controls
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
