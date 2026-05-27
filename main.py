import ollama
import json
import pandas as pd
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "data" / "All_Recipe_Web_Scraping_Dataset_Labeled.csv"
CUISINE_TYPES = ["Italian", "Chinese", "Mexican", "Indian", "American"]


def load_dataset():
    try:
        df = pd.read_csv(DATASET_PATH)
        df.set_index("Name", inplace=True)
        return df
    except FileNotFoundError:
        print(f"Warning: Dataset not found at {DATASET_PATH}. Running without context.")
        return None


def lookup_context(dish_name, dataset):
    if dataset is None:
        return None, None
    try:
        row = dataset.loc[dish_name]
        ingredients = str(row["Ingredients"]).replace("|", ", ") if pd.notna(row.get("Ingredients")) else None
        description = row.get("Description") if pd.notna(row.get("Description")) else None
        return ingredients, description
    except KeyError:
        print(f"Note: '{dish_name}' not found in dataset. Running with dish name only.")
        return None, None


# TODO: try changing this prompt to see how the output changes
PROMPT = """
Classify the following recipe into its cuisine type.

Dish name: {dish_name}
{context}
Return a JSON object with exactly these fields:
- cuisine_type: the cuisine type
- confidence_score: integer from 0 to 100
- reasoning: brief explanation of why this cuisine type fits
"""

dataset = load_dataset()
dish_name = input("Enter dish name: ")

ingredients, description = lookup_context(dish_name, dataset)

context_lines = []
if ingredients:
    context_lines.append(f"Ingredients: {ingredients}")
if description:
    context_lines.append(f"Description: {description}")
context_str = "\n".join(context_lines) + "\n" if context_lines else ""

result = ollama.generate(
    model="gemma2:2b",
    prompt=PROMPT.format(dish_name=dish_name, context=context_str),
    system="You are a culinary expert who classifies recipes into exactly one of these five cuisine types: Italian, Chinese, Mexican, Indian, American. Use the dish name, ingredients, and description to make your decision. Always return valid JSON with cuisine_type, confidence_score, and reasoning.",
    format="json",
)

output = json.loads(result.response)
print(json.dumps(output, indent=2))
