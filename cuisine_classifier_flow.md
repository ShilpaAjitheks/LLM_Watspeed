# Cuisine Classifier — Exact System Flow

---

# Streamlit UI Flow (`app.py`)

Entry point: `uv run streamlit run app.py`

---

## On App Load (cached, runs once)

- `init()` → loads `config.yaml` + CSV dataset into memory (`@st.cache_resource`)
- `adapter_available()` → checks Ollama for `cuisine-classifier:latest`

---

## UI Inputs (rendered top to bottom)

1. **Dish name** — text input
2. **Recipe card image** — optional file uploader (png/jpg)
3. **Model** — radio: `Base model` / `Adapter (Option B)` / `Compare both`
4. **Checkboxes** — `Look it up in recipe dataset` / `Use retrieval-augmented few-shot` / `Use cuisine knowledge RAG`
5. **Embedding model** — radio (only shown if retrieval is checked): `nomic-embed-text` or `fine-tuned mpnet`

---

## Image Upload Flow (runs immediately on upload, before Classify)

1. Show image preview
2. Call `extract_recipe_raw(tmp_path)` → vision model (`llava:7b`) extracts raw JSON
3. `parse_extraction(raw)` → parse the JSON; if it fails, retry with `gemma2:2b`
4. `check_extraction()` → structural check (dish_name, ingredients, description present) + semantic plausibility check
5. Result cached in `st.session_state` keyed by `(filename, size)` — not re-run on re-render
6. If PASS → `image_ingredients` and `image_description` are set and used instead of CSV
7. If FAIL → show error; warn user to remove image or try another

---

## On "Classify" Button Click

### Single model (`Base model` or `Adapter`)
1. Call `run_classify(model_name)` → calls `predict_cuisine()` (same core function as CLI)
2. Pass `show_result()` the returned `(result, prompt, system_prompt, context_found, few_shot_examples)`

### Compare both
1. Render two columns side-by-side (visual layout only)
2. Run `run_classify()` for base model in left column — **blocks until complete**
3. Run `run_classify()` for adapter in right column — starts only after base model finishes
4. Both are **sequential**, not parallel (`st.columns` is layout only, no threading)

---

## `show_result()` — What Gets Displayed

| Element | Shown when |
|---|---|
| Mode label (e.g. "dataset context + retrieval few-shot") | Always |
| **Cuisine** heading + confidence progress bar | Always |
| Reasoning bullet points | Always |
| Ambiguity badge (info / warning / error) | RAG was used |
| "Retrieved cuisine cards" expander | RAG was used |
| "Retrieved few-shot examples" expander | Retrieval was used |
| "Prompt sent to LLM" expander (system + user) | Always |

### Mode Label — How It Is Built

Parts are joined with ` + `; if no parts apply, shows `"zero-shot"`.

| Condition | Label part added |
|---|---|
| Image extraction succeeded | `"image extraction"` |
| Dataset checked + dish found | `"dataset context"` |
| Dataset checked + dish NOT found | `"dish name only (not found in dataset)"` |
| Retrieval checkbox on | `"retrieval few-shot (mpnet)"` or `"retrieval few-shot (nomic-embed-text)"` |
| Knowledge RAG checkbox on | `"cuisine knowledge RAG"` |
| Adapter model used | `"LoRA adapter"` |
| None of the above | `"zero-shot"` |

Example combinations:
- `dataset context`
- `dataset context + retrieval few-shot (mpnet)`
- `dataset context + cuisine knowledge RAG`
- `image extraction + LoRA adapter`
- `zero-shot`

---

## Key Difference vs CLI

The Streamlit app calls the exact same `predict_cuisine()` function from `__main__.py` — the UI is purely a wrapper. The only Streamlit-specific logic is:
- Image extraction happens eagerly on upload (not at classify time)
- Results from both models can be shown side-by-side in "Compare both" mode (but run sequentially)
- All heavy resources (config, dataset, vector stores) are `@st.cache_resource` — loaded once per session

---

# CLI Flow (`__main__.py`)

Entry point: `uv run python -m cuisine_classifier "<dish name>" [flags]`

---

## Step 1 — Input & Config

- Parse CLI args: dish name, `--rag`, `--retrieval`, `--image`, `--no-context`, `--embed-model`
- Load `config.yaml` (model name, temperature, dataset path, adapter name)

---

## Step 2 — Context Acquisition (choose one path)

### Path A — Image extraction (`--image`)
1. `vision.extract_from_image(path)` calls a vision LLM to pull ingredients + description from a recipe card image
2. `vision.check_extraction()` quality-checks the result
3. If PASS → use extracted ingredients/description; if FAIL → fall back to CSV lookup

### Path B — CSV dataset lookup (default)
1. Load `data/All_Recipe_Web_Scraping_Dataset_Labeled.csv` (13,057 recipes, indexed by Name)
2. Look up the dish name → retrieve `Ingredients` and `Description` columns
3. If not found → proceed with dish name only

---

## Step 3 — Few-shot Retrieval (optional, `--retrieval`)

- Query ChromaDB `VectorStore` with `dish_name + ingredients + description` as the query
- Retrieve k=4 nearest neighbours; filter out the dish itself; keep top 3
- Embed model: `mpnet` (default) or `nomic-embed-text`
- These 3 examples are appended to the prompt as dynamic few-shot examples

---

## Step 4 — RAG (optional, `--rag`, base model only)

1. **Ambiguity signal**: query ChromaDB with ingredients+description, get k=3 neighbours; check if they all agree (confident), split 2-way (mild), or 3-way (high)
2. **Knowledge cards**: `retrieval/knowledge_store.retrieve_cuisine_cards()` returns structured cuisine rule cards for the dish
3. Builds a `build_rag_prompt()` instead of the standard prompt — includes ambiguity level + cuisine rule cards

---

## Step 5 — Prompt Construction

### Standard path (base model, no RAG)
`build_prompt(dish_name, ingredients, description, few_shot_examples)`:
- Prepends `_STATIC_FEW_SHOT` (20 hand-crafted disambiguation examples)
- Appends dynamic few-shot examples if retrieval was used
- Adds format-override rules (e.g. "egg rolls → Chinese", "taco → Mexican")
- Appends dish name + ingredients + description

### RAG path
`build_rag_prompt()`:
- Includes ambiguity level + note
- Includes retrieved cuisine knowledge cards (format rules, traps, confused-with)
- Uses `_RAG_SYSTEM_PROMPT` (shorter, card-focused)

---

## Step 6 — LLM Inference (Ollama)

### Base model path (`ollama.chat`)
- System prompt: long detailed system prompt with 17 decision rules + format overrides
- User message: the constructed prompt
- Structured output via `format=CuisineResponse.model_json_schema()` (Pydantic)
- Response validated into `CuisineResponse(Recipe_name, Reasoning, Confidence_score, Cuisine)`

### Adapter/fine-tuned model path (`ollama.generate`)
- Manually applies Gemma2 chat template (`<bos><start_of_turn>user\n...<end_of_turn>`)
- Merges system prompt + user prompt into one string
- Appends JSON format instruction
- Response is free-form text → parsed by `_parse_free_form()` with regex fallbacks

---

## Step 7 — Output

- Returns dict: `{Recipe_name, Reasoning, Confidence_score, Cuisine}`
- If RAG was used: also includes `ambiguity` and `retrieved_cards` fields
- Printed as pretty JSON to stdout

---

## CLI Modes Summary

| Flag | Mode name | What changes |
|---|---|---|
| (none) | context-enhanced | CSV lookup only |
| `--no-context` | baseline | dish name only, no CSV |
| `--retrieval` | few-shot retrieval | + ChromaDB dynamic examples |
| `--rag` | cuisine-knowledge-rag | + ambiguity signal + knowledge cards |
| `--image` | image-extraction | vision LLM replaces CSV lookup |

---

## Classification Labels

`American | Italian | Indian | Mexican | Chinese | Other`

Key rules:
- American is the **residual/default** for this US recipe dataset
- Other = only identifiable foreign cuisines outside the 5 (French, Korean, Thai, etc.)
- Format words override ingredients (e.g. "tiramisu" → Italian regardless of filling)
- Substitute proteins inherit the original dish's cuisine
