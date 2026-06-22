#!/usr/bin/env python3
"""
Cuisine Type Classifier

Classifies a recipe into one of: Italian, Chinese, Mexican, Indian, American.
Looks up ingredients and description from the dataset by dish name.
Falls back to dish name only if the dish is not found.
"""

import sys
import json
import re
import yaml
import ollama
import pandas as pd
from pathlib import Path
from typing import Literal
from pydantic import BaseModel


class CuisineResponse(BaseModel):
    recipe_name: str
    thoughts: list[str]
    confidence_score: int
    cuisine: Literal["American", "Italian", "Indian", "Mexican", "Chinese", "Other"]


def load_config(config_path=None):
    if config_path is None:
        import os
        package_dir = os.path.dirname(__file__)
        config_path = os.path.join(package_dir, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df.set_index("Name", inplace=True)
        return df
    except FileNotFoundError:
        print(f"Warning: Dataset not found at {csv_path}. Running without context.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Error loading dataset: {e}", file=sys.stderr)
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
        print(f"Note: '{dish_name}' not found in dataset. Running with dish name only.", file=sys.stderr)
        return None, None


_STATIC_FEW_SHOT = (
    "Use these examples to calibrate where the cuisine boundaries fall.\n"
    "Judge by core ingredients and origin, not the dish's English name:\n\n"
    "Mom's Chicken and Dumplings    -> American  (biscuit-dough dumplings, not Chinese)\n"
    "The Ultimate Pasta Salad       -> American  (pasta in name, US picnic dish)\n"
    "Italian Wedding Cookies        -> American  ('Italian' name, US butter-almond cookie)\n"
    "Baked Garlic Parmesan Chicken  -> American  (breadcrumb-Parmesan bake is US weeknight cooking — no tomato sauce or mozzarella, NOT Italian Parmigiana)\n"
    "Chef John's Meatless Meatballs -> Italian   (substitute protein inherits the dish)\n"
    "Tofu Tacos                     -> Mexican   (substitute protein inherits the dish)\n"
    "Baked BBQ Pulled Pork Nachos   -> Mexican   (nacho base is Mexican even with BBQ)\n"
    "Navajo Tacos                   -> American  (fry-bread taco is US, NOT Mexican)\n"
    "Authentic Saag Paneer          -> Indian    (paneer + spinach + fenugreek)\n"
    "Authentic Chicken Tikka Masala -> Indian    (yogurt marinade + garam masala)\n"
    "Char Siu (Chinese BBQ Pork)    -> Chinese   ('BBQ' name, but hoisin/five-spice)\n"
    "Chicken Lettuce Wraps          -> Chinese   (hoisin/soy stir-fried filling)\n"
    "Yaki Mandu (Korean Dumplings)  -> Other     (Korean — outside the five)\n"
    "Sunomono (Japanese Salad)      -> Other     (Japanese — outside the five)\n"
    "Pho Bo (Vietnamese Beef Soup)  -> Other     (Vietnamese — outside the five)\n"
    "French Onion Soup              -> Other     (French — outside the five)\n"
)


def build_prompt(dish_name, ingredients=None, description=None, few_shot_examples=None):
    context_lines = []
    if ingredients:
        context_lines.append(f"Ingredients: {ingredients}")
    if description:
        context_lines.append(f"Description: {description}")
    context_str = "\n".join(context_lines) + "\n" if context_lines else ""

    dynamic_str = ""
    if few_shot_examples:
        lines = ["\nAdditional similar recipes retrieved for this dish:"]
        for i, ex in enumerate(few_shot_examples, 1):
            lines.append(f"  Example {i}: {ex['dish_name']} → {ex['cuisine_type']}")
        dynamic_str = "\n".join(lines) + "\n"

    return f"""{_STATIC_FEW_SHOT}{dynamic_str}
Now classify the following recipe.

Dish name: {dish_name}
{context_str}"""


def predict_cuisine(dish_name, config, dataset=None, use_retrieval=False, vector_store=None):
    cuisine_types = config["cuisine_types"]
    system_prompt = (
        "You are a culinary expert classifying a dish into exactly ONE cuisine.\n"
        "Categories: American, Italian, Indian, Mexican, Chinese, Other.\n\n"
        "Think step by step in 2-3 short lines: explicit cuisine claim? "
        "dish format owned by one cuisine? what does the ingredient pantry suggest?\n\n"
        "CLASSIFICATION GUIDE:\n"
        "Italian — pasta/pizza/risotto with Italian sauces, cheeses or meats; Italian-American classics "
        "(chicken parmigiana, baked ziti, stuffed shells, biscotti); pasta in Italian sauce even with "
        "non-Italian proteins. Exception: if pasta is just the vehicle for a non-Italian flavor profile "
        "(Cajun shrimp pasta, fajita pasta) → classify by that flavor profile, not Italian.\n"
        "Mexican — key markers: tomatillo, cotija, chipotle, jalapeño, corn/flour tortilla, mole, pozole, "
        "salsa, avocado as primary ingredient. Tacos/quesadillas/enchiladas → Mexican even with non-traditional proteins.\n"
        "American — US regional cuisines (Southern, Cajun/Creole, New England, Tex-Mex); generic comfort food "
        "with no ethnic markers (stews, casseroles, baked goods using standard US pantry: condensed soup, "
        "cream cheese, butter, all-purpose flour).\n\n"
        "OTHER RULE: The Other category covers ALL cuisines outside the five main ones: "
        "Japanese, Korean, Vietnamese, Thai, French, German, Greek, Spanish, "
        "Middle Eastern, South American (Brazilian, Peruvian, Colombian), African, Caribbean, etc. "
        "Do NOT classify French, German, Greek, or other European dishes as Italian. "
        "Do NOT classify Japanese, Korean, or Vietnamese dishes as Chinese. "
        "Use Other ONLY for a cuisine clearly outside the five — never pick Other just because "
        "you are unsure between two of the five; pick the most probable of the five instead."
    )

    print(f"[1/4] Dataset lookup: fetching context for '{dish_name}'...")
    ingredients, description = lookup_context(dish_name, dataset)
    context_found = bool(ingredients or description)
    if context_found:
        print(f"[1/4] Dataset lookup: context retrieved successfully.")
    else:
        print(f"[1/4] Dataset lookup: no context found — proceeding with dish name only.")

    few_shot_examples = None
    if use_retrieval and vector_store is not None:
        query_text = " ".join(filter(None, [ingredients, description]))
        if not query_text:
            query_text = dish_name
        candidates = vector_store.query(query_text, k=4)
        few_shot_examples = [ex for ex in candidates if ex["dish_name"] != dish_name][:3]
        print(f"[2/4] ChromaDB retrieval: {len(few_shot_examples)} few-shot examples retrieved ({vector_store.model}).")
    else:
        print(f"[2/4] ChromaDB retrieval: skipped — zero-shot mode.")

    prompt = build_prompt(dish_name, ingredients, description, few_shot_examples)

    print(f"[3/4] Sending prompt to {config['model']['name']}...")
    try:
        result = ollama.chat(
            model=config["model"]["name"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            format=CuisineResponse.model_json_schema(),
            options={"temperature": config["model"]["temperature"]},
        )
        parsed = CuisineResponse.model_validate_json(result.message.content)
        output = parsed.model_dump()

        if output["cuisine"] not in cuisine_types:
            print(f"Note: model returned '{output['cuisine']}' — remapped to 'Other'.", file=sys.stderr)
            output["cuisine"] = "Other"

        print(f"[4/4] Response received — classification complete.")
        return output, prompt, system_prompt, context_found, few_shot_examples

    except Exception as e:
        print(f"Error calling Ollama: {e}", file=sys.stderr)
        return None, prompt, system_prompt, context_found, few_shot_examples


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Classify recipe cuisine type from dish name")
    parser.add_argument("dish_name", type=str, help="Name of the dish to classify")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--no-context", action="store_true", help="Disable dataset lookup (dish name only)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to recipe dataset CSV (overrides config)")

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found.", file=sys.stderr)
        sys.exit(1)

    dataset = None
    if not args.no_context:
        import os
        if args.dataset:
            dataset_path = args.dataset
        else:
            package_dir = os.path.dirname(__file__)
            project_root = os.path.dirname(os.path.dirname(package_dir))
            dataset_path = os.path.join(project_root, config["dataset"]["path"])

        dataset = load_dataset(dataset_path)
        if dataset is not None:
            print(f"Loaded dataset with {len(dataset)} recipes")

    print(f"Classifying: {args.dish_name}")
    print(f"Using model: {config['model']['name']}")
    mode = "baseline" if args.no_context or dataset is None else "context-enhanced"
    print(f"Mode: {mode}")
    print()

    output, _, __, ___, ____ = predict_cuisine(args.dish_name, config, dataset=dataset)

    if output:
        print(json.dumps(output, indent=2))
    else:
        print("Failed to classify cuisine.")
        sys.exit(1)


if __name__ == "__main__":
    main()
