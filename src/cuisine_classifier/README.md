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
2. Looks up ingredients and description for the given dish
3. Builds a prompt with available context and sends it to `gemma2:2b`
4. Returns structured JSON with cuisine type, confidence score, and reasoning
5. Falls back to dish name only if the dish is not found in the dataset
6. Remaps any out-of-scope cuisine labels to `Other`

## Project Structure

```
src/cuisine_classifier/
├── __init__.py       # Package declaration
├── __main__.py       # Classifier logic and CLI entry point
├── config.yaml       # Model settings and cuisine types
└── README.md

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
