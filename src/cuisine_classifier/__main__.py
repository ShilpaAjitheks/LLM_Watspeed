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
REMINDER — these format rules override ingredients:
- dish name contains fried rice / egg rolls / wontons / lo mein / dumplings → Chinese
- dish name contains garlic bread / bruschetta / calzone / biscotti / tiramisu / lasagna / pizza dough → Italian
- dish name contains taco / enchilada / burrito / quesadilla / nacho / tostada / fajita → Mexican (unless fry bread or Navajo)
- dish name contains Russian / Swedish / Danish / Polish / Korean / Vietnamese / Japanese / Thai / Filipino / Greek / French / German / Middle Eastern / Cuban / Brazilian / Peruvian / Caribbean / Jamaican / African / Irish / British → Other
- generic everyday drink / snack / dessert / baked good with no foreign signal → American

Dish name: {dish_name}
{context_str}"""


def predict_cuisine(dish_name, config, dataset=None, use_retrieval=False, vector_store=None):
    cuisine_types = config["cuisine_types"]
    system_prompt = (
        "You are an expert cuisine classifier working on a recipe dataset from AllRecipes.com.\n"
        "Assign each recipe to EXACTLY ONE of these six cuisine categories:\n\n"
        "  American | Italian | Indian | Mexican | Chinese | Other\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "CATEGORY DEFINITIONS\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "AMERICAN\n"
        "  Dishes rooted in United States culinary traditions, including all US regional styles.\n"
        "  Includes: Southern, Cajun/Creole, New England, Tex-Mex, Midwestern comfort food, American BBQ.\n"
        "  Includes: French Toast, German Chocolate Cake, Buffalo wings — despite misleading names.\n"
        "  Includes: American baked goods (muffins, cheesecake, brownies, banana bread, apple pie).\n"
        "  Includes: Generic US home cooking — pot roast, pork tenderloin, sheet-pan dinners,\n"
        "    egg/breakfast casseroles, meatloaf, beef jerky, coconut macaroons, Southern sides\n"
        "    (collard greens, cornbread). Default here when no other cuisine signal is present.\n\n"
        "ITALIAN\n"
        "  Dishes with Italian culinary origins: pasta, pizza, risotto, Italian sauces/cheeses/meats.\n"
        "  Includes Italian-American classics that kept the Italian dish identity (chicken parmesan, baked ziti).\n"
        "  Pasta in a clearly Italian sauce (carbonara, bolognese, marinara, pesto) → Italian even with non-Italian protein.\n"
        "  Exception: pasta in a Cajun, Tex-Mex, or otherwise American sauce → American.\n\n"
        "INDIAN\n"
        "  Dishes from the Indian subcontinent (India, Pakistan, Bangladesh, Sri Lanka).\n"
        "  Includes regional Indian styles: Punjabi, Bengali, South Indian, Hyderabadi, Mughlai.\n"
        "  Chicken Tikka Masala → Indian (classify by dish tradition, not debate).\n\n"
        "MEXICAN\n"
        "  Dishes from Mexican culinary tradition, including Tex-Mex.\n"
        "  Tex-Mex (nachos, hard-shell tacos, fajitas, queso dip) → Mexican.\n"
        "  Ceviche in a Mexican context → Mexican.\n"
        "  Key markers: green chile, huitlacoche, tomatillo, street tacos, quesadillas,\n"
        "    guacamole, enchilada/salsa-verde flavors. A casserole built on these → Mexican.\n"
        "  IMPERATIVE: any dish named around quesadilla, nacho, taco, burrito, enchilada,\n"
        "  tostada, or fajita → Mexican, regardless of modifiers (15-minute, supreme,\n"
        "  Impossible, easy) or exotic ingredients. Modifiers never override this.\n\n"
        "CHINESE\n"
        "  Dishes from Chinese culinary tradition, including American-Chinese restaurant food.\n"
        "  General Tso's, Orange Chicken, Beef with Broccoli → Chinese.\n"
        "  Broad 'Asian' stir-fries clearly Chinese in style → Chinese.\n"
        "  Japanese, Korean, Thai, Vietnamese, Filipino → Other (not Chinese).\n\n"
        "OTHER\n"
        "  Any cuisine not in the five above.\n"
        "  Examples: French, Japanese, Korean, Thai, Vietnamese, Filipino, Greek, Middle Eastern, "
        "German, Swedish, Danish, Russian, Polish, Cuban, Brazilian, Peruvian, Caribbean/Jamaican, "
        "African, British/Irish.\n"
        "  Also: fusion dishes with no clear dominant cuisine.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "DECISION RULES\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "0. DEFAULT RULE — American is the RESIDUAL category for this US recipe dataset.\n"
        "   ALL everyday desserts, baked goods, cookies, doughnuts, cakes, pies, snacks,\n"
        "   chicken nuggets, beverages/drinks, lemonades, sauces, applesauce, and generic\n"
        "   home cooking → American, UNLESS the dish clearly belongs to one of the other\n"
        "   five cuisines or a specific foreign cuisine. NEVER put a generic dessert,\n"
        "   drink, or snack in Other. Other is ONLY for identifiable non-listed foreign\n"
        "   cuisines (French, Thai, Korean, Greek, etc.).\n"
        "1. Classify by DISH CONCEPT, not individual ingredients (garlic, olive oil, tomatoes are everywhere).\n"
        "2. Misleading place names: French Toast → American; German Chocolate Cake → American; Scotch Eggs → Other.\n"
        "3. Cajun / Creole → always American.\n"
        "4. Tex-Mex → always Mexican.\n"
        "5. American-Chinese restaurant food → always Chinese.\n"
        "6a. Pasta in Italian sauce → Italian; pasta in American-style sauce → American.\n"
        "6b. Plant-based / substitute-protein versions (Beyond, Impossible, vegan, tofu swaps) inherit the cuisine "
        "of the ORIGINAL dish — Beyond spaghetti & meatballs → Italian; Impossible street tacos → Mexican.\n"
        "7. Use Other ONLY for dishes that clearly belong to a cuisine OUTSIDE the five "
        "(French, Japanese, Korean, Thai, Vietnamese, Filipino, Greek, Middle Eastern, German, "
        "Swedish, Danish, Russian, Polish, Cuban, Brazilian, Peruvian, Caribbean/Jamaican, African, British/Irish, etc.).\n"
        "   Do NOT use Other because you are unsure between two of the five — pick the MOST PROBABLE of "
        "{American, Italian, Indian, Mexican, Chinese}.\n"
        "   A dish with no foreign-cuisine signal defaults to American, never Other.\n"
        "8.  Hawaiian → American (Hawaii is a US state): kalua pork, Hawaiian rolls/sliders, Hawaiian coleslaw.\n"
        "9.  Goulash → American (macaroni-tomato-beef US style) UNLESS explicitly 'Hungarian'.\n"
        "10. Stroganoff → American (ground/beef stroganoff with egg noodles) UNLESS explicitly 'Russian'.\n"
        "11. Generic US dishes default to American: sliders, sheet-pan dinners, dump/slow-cooker casseroles,\n"
        "    Reuben, corned beef — when no other cuisine is signalled.\n"
        "12. Pasta routing: mac and cheese, pasta salad, and buffalo / ranch / BBQ / taco pasta → American\n"
        "    (pasta is just the vehicle). Alfredo, cacciatore, piccata, marsala, lasagna → Italian.\n"
        "13. 'Curry' alone is NOT Indian. Thai / coconut-lemongrass curry, Japanese curry, Caribbean curry → Other.\n"
        "    Only subcontinental masala / ghee / garam-masala curries → Indian.\n"
        "14. Latin America is NOT Mexican: Cuban, Brazilian, Peruvian, Argentine, Puerto Rican, empanadas,\n"
        "    chimichurri → Other. Ceviche → Other unless the context is explicitly Mexican.\n"
        "15. Mexican casseroles count: enchilada / burrito / taco casseroles and layered Mexican dips → Mexican,\n"
        "    even when named 'easy' or 'casserole'.\n"
        "16. Unmarked stir-fry, fried rice, lo mein, egg rolls, dumplings → Chinese by default.\n"
        "    If marked Thai / Vietnamese / Korean / Japanese, the modifier wins → Other\n"
        "    (Kimchi Fried Rice → Other; Yellow Curry Fried Rice → Other). Taiwanese → Chinese.\n"
        "17. Other = any cuisine outside the five, e.g. French, Japanese, Korean, Thai, Vietnamese, Filipino,\n"
        "    Greek, Middle Eastern, German, Swedish, Danish, Russian, Polish, Cuban, Brazilian, Peruvian,\n"
        "    Caribbean / Jamaican, African, British / Irish (shepherd's pie, soda bread). NOTE: Hawaiian is the\n"
        "    exception — it goes to American (rule 8), not Other.\n\n"
        "IMPERATIVE DISH FORMAT OVERRIDES — ingredients never override these:\n"
        "  Chinese formats: fried rice / egg rolls / wontons / lo mein / stir-fry in Chinese style / dumplings\n"
        "    → Chinese REGARDLESS of protein or modifier.\n"
        "    (Turkey Fried Rice → Chinese; Keto Beef Egg Roll Slaw → Chinese; Local Kine Wontons → Chinese)\n"
        "  Italian formats: garlic bread / bruschetta / calzone / biscotti / tiramisu / lasagna / pizza dough /\n"
        "    risotto / carbonara / bolognese / pesto pasta\n"
        "    → Italian REGARDLESS of modifier or dietary swap.\n"
        "    (Gluten-Free Biscotti → Italian; Twinkie Tiramisu → Italian; Whole Wheat Pizza Dough → Italian;\n"
        "     Grandma's Sour Cream Lasagna → Italian; Meatball-Stuffed Garlic Bread → Italian)\n\n"
        "EXCEPTION TO DEFAULT RULE — do NOT classify as American when the dish name carries an explicit\n"
        "foreign nationality marker: Russian, Swedish, Danish, Polish, Korean, Vietnamese, Japanese, Thai, Filipino, "
        "Greek, French, German, Middle Eastern, Cuban, Brazilian, Peruvian, Caribbean, Jamaican, African, Irish, British, etc.\n"
        "  Russian Ant Hill Cake → Other; Swedish Meatballs → Other; Homemade Pita Bread → Other;\n"
        "  Korean Barbecue-Style Meatballs → Other; French Onion Soup → Other.\n"
        "  The default-to-American rule applies only when NO foreign signal is present.\n\n"
        "DESCRIPTION / SERVING CONTEXT WARNING: US cultural references in the description — Super Bowl,\n"
        "Halloween, Christmas, Thanksgiving, tailgate, holiday party — tell you WHERE the dish is served,\n"
        "NOT what cuisine it is. Do NOT use them to classify as American. A calzone shaped like a snake\n"
        "for Halloween is still Italian; garlic bread sliders served at a Super Bowl party are still Italian.\n\n"
        "ADDITIONAL MARKER INGREDIENTS:\n"
        "Italian: mascarpone cheese → strong Italian marker (tiramisu, cannoli, panna cotta).\n"
        "Other/Korean: gochujang (Korean hot pepper paste) → Korean → Other.\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "REASONING ORDER\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Follow this order strictly:\n"
        "1. Dish name — form a hypothesis. If the name unambiguously signals a cuisine "
        "(biscotti, wontons, enchiladas, tiramisu, bruschetta), start there.\n"
        "2. Marker ingredients — confirm or override the hypothesis. If distinctive cuisine markers "
        "clearly contradict the dish name, the ingredients win (fusion case). "
        "Generic ingredients (flour, butter, eggs, oil, salt, water) carry NO cuisine signal — "
        "never let them push you toward American or away from the dish name.\n"
        "3. Description context — use ONLY for explicit cuisine claims or foreign technique mentions. "
        "Ignore US cultural references (Super Bowl, Halloween, Christmas) — they are serving context, not cuisine.\n"
        "Use Other ONLY when the dish clearly belongs to a cuisine outside the five (French, Thai, Korean, Greek, etc.).\n"
        "Never use any label outside: American, Italian, Indian, Mexican, Chinese, Other."
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
