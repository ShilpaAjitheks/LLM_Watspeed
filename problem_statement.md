# Problem Statement

## Problem Context

The provided recipe dataset contains over 13,000 recipes with dish names, ingredients, and descriptions, but no cuisine type labels. Knowing the cuisine type of a recipe is useful for filtering, recommendation, and organizing recipe collections. An LLM can infer cuisine type from the combination of dish name, ingredients, and description without needing explicit labels in the data. Ingredients and description are automatically looked up from the dataset by dish name, following the same pattern as the baseline recipe classifier.

## Problem Statement

Given a dish name, classify it into one of five cuisine types: Italian, Chinese, Mexican, Indian, or American. Ingredients and description are auto-fetched from the dataset when available; if the dish is not in the dataset, the classification falls back to dish name only.

## LLM Role

Classification — the LLM picks one label from a fixed set of five cuisine categories based on the recipe inputs.

## Example

**Flow 1 — Dish found in dataset (auto-context):**
- Input: `Vegetable Quesadillas` (exists in dataset)
- Dataset auto-fetches ingredients and description
- Expected Output:
```json
{
  "cuisine_type": "Mexican",
  "confidence_score": 95,
  "reasoning": "Quesadillas are a classic Mexican dish. The ingredients like tortillas, cheese, and vegetables common in Mexican cuisine strongly suggest this is a Mexican recipe."
}
```

**Flow 2 — Dish not in dataset (fallback to dish name only):**
- Input: `Butter Chicken` (not in dataset)
- No context fetched — LLM classifies from dish name alone
- Expected Output:
```json
{
  "cuisine_type": "Indian",
  "confidence_score": 95,
  "reasoning": "Butter Chicken is a classic North Indian dish characterized by its creamy tomato-based gravy with fragrant spices."
}
```

## Success Criteria

The LLM returns valid JSON with the three required fields. The `cuisine_type` value is one of the five defined categories. For dataset-found dishes, the `reasoning` references the auto-fetched ingredients or description. For fallback dishes, classification is still correct based on dish name alone.
