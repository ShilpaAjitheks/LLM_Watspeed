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


def build_prompt(dish_name, ingredients=None, description=None):
    context_lines = []
    if ingredients:
        context_lines.append(f"Ingredients: {ingredients}")
    if description:
        context_lines.append(f"Description: {description}")
    context_str = "\n".join(context_lines) + "\n" if context_lines else ""

    # TODO: try changing this prompt to see how the output changes
    return f"""Classify the following recipe into its cuisine type.

Dish name: {dish_name}
{context_str}
Return a JSON object with exactly these fields:
- cuisine_type: the cuisine type
- confidence_score: integer from 0 to 100
- reasoning: brief explanation of why this cuisine type fits
"""


def predict_cuisine(dish_name, config, dataset=None):
    cuisine_types = config["cuisine_types"]
    allowed = ", ".join(cuisine_types)
    system_prompt = (
        f"You are a culinary expert who classifies recipes into exactly one of these cuisine types: {allowed}. "
        "If the dish does not clearly belong to Italian, Chinese, Mexican, Indian, or American cuisine, you MUST use 'Other'. "
        "Never invent a cuisine type outside this list. "
        "Always return valid JSON with cuisine_type, confidence_score, and reasoning."
    )

    ingredients, description = lookup_context(dish_name, dataset)
    prompt = build_prompt(dish_name, ingredients, description)

    try:
        result = ollama.generate(
            model=config["model"]["name"],
            prompt=prompt,
            system=system_prompt,
            format="json",
            options={"temperature": config["model"]["temperature"]},
        )
        output = json.loads(result.response)

        # If model returns a type outside the allowed list, remap to Other
        if output.get("cuisine_type") not in cuisine_types:
            print(f"Note: model returned '{output.get('cuisine_type')}' — remapped to 'Other'.", file=sys.stderr)
            output["cuisine_type"] = "Other"

        return output

    except Exception as e:
        print(f"Error calling Ollama: {e}", file=sys.stderr)
        return None


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

    output = predict_cuisine(args.dish_name, config, dataset=dataset)

    if output:
        print(json.dumps(output, indent=2))
    else:
        print("Failed to classify cuisine.")
        sys.exit(1)


if __name__ == "__main__":
    main()
