import base64
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import ollama

VISION_MODEL = "llava:7b"     # multimodal — reads images (gemma4:e4b crashes on Windows)
TEXT_MODEL   = "gemma2:2b"    # text-only — used for correction

EXTRACTION_PROMPT = """Extract the recipe from this recipe card image.
Return JSON only — no explanation, no markdown fences.
Required fields:
  dish_name: string
  ingredients: list of strings (one item per bullet)
  description: string (one or two sentences describing the dish)"""

EXPECTED_SCHEMA = """{
  "dish_name": "string",
  "ingredients": ["string", "string", "..."],
  "description": "string"
}"""


def load_image_b64(image_path: str) -> str:
    """Load an image file and return as base64 string for Ollama."""
    return base64.b64encode(Path(image_path).read_bytes()).decode()


def extract_recipe_raw(image_path: str) -> str:
    """Call gemma4:e4b with the image and return the raw text response."""
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role":    "user",
            "content": EXTRACTION_PROMPT,
            "images":  [load_image_b64(image_path)],
        }],
        options={"temperature": 0},
    )
    return response.message.content.strip()


def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text.strip(), flags=re.MULTILINE)
    return text.strip()


def correct_with_model(malformed_text: str) -> str:
    """Ask gemma2:2b to reformat malformed extraction output to match the schema.

    The corrector never sees the image — it only does schema repair on text.
    """
    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{
            "role":    "user",
            "content": (
                f"The following text should be valid JSON matching this schema:\n{EXPECTED_SCHEMA}\n\n"
                f"Reformat it to match exactly. Return JSON only — no explanation.\n\n"
                f"Text to reformat:\n{malformed_text}"
            ),
        }],
        options={"temperature": 0},
    )
    return response.message.content.strip()


def parse_extraction(raw_text: str, max_retries: int = 1) -> Optional[dict]:
    """Parse raw model output to dict. Attempt correction if parsing fails.

    Step 1: strip markdown fences (most common failure mode)
    Step 2: if still invalid, ask corrector model to reformat
    Step 3: if still invalid after retries, return None and surface raw text
    """
    cleaned = strip_markdown_fences(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for attempt in range(max_retries):
        print(f"  JSON parse failed — asking corrector model (attempt {attempt + 1})...")
        corrected = strip_markdown_fences(correct_with_model(cleaned))
        try:
            return json.loads(corrected)
        except json.JSONDecodeError:
            cleaned = corrected

    print("  Could not parse after correction. Raw output returned for inspection:")
    print(raw_text[:400])
    return None


def extract_from_image(image_path: str) -> Optional[dict]:
    """Full extraction pipeline: vision model → parse → correct if needed."""
    raw = extract_recipe_raw(image_path)
    return parse_extraction(raw)


def check_extraction(extracted: Optional[dict], typed_dish_name: str) -> dict:
    """Verify extracted recipe against typed dish name. Returns a pass/fail report.

    Structural completeness: all required fields present with correct types.
    Semantic plausibility: dish name matches typed input, sufficient ingredients,
    non-empty description.
    dish_name_mismatch is a warning flag — it does not cause a hard fail.
    """
    if extracted is None:
        return {
            "passed": False,
            "issues": ["extraction returned None — parse failed"],
            "dish_name_mismatch": False,
            "extracted_dish_name": "",
        }

    issues = []

    # Structural completeness
    for field in ("dish_name", "ingredients", "description"):
        if field not in extracted:
            issues.append(f"missing field: {field}")
        elif field == "ingredients" and not isinstance(extracted[field], list):
            issues.append(f"ingredients is {type(extracted[field]).__name__}, expected list")
        elif field == "description" and not isinstance(extracted[field], str):
            issues.append(f"description is {type(extracted[field]).__name__}, expected str")

    if issues:
        return {
            "passed": False,
            "issues": issues,
            "dish_name_mismatch": False,
            "extracted_dish_name": extracted.get("dish_name", ""),
        }

    # Semantic plausibility
    if len(extracted.get("ingredients", [])) < 3:
        issues.append(
            f"too few ingredients: got {len(extracted['ingredients'])}, expected >= 3"
        )
    if not extracted.get("description", "").strip():
        issues.append("description is empty")

    extracted_dish = extracted.get("dish_name", "")
    if typed_dish_name and extracted_dish:
        similarity = SequenceMatcher(
            None, typed_dish_name.lower(), extracted_dish.lower()
        ).ratio()
        dish_name_mismatch = similarity < 0.75
    else:
        dish_name_mismatch = False

    if dish_name_mismatch:
        issues.append(
            f"dish name mismatch: image shows '{extracted_dish}', expected '{typed_dish_name}'"
        )

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "dish_name_mismatch": dish_name_mismatch,
        "extracted_dish_name": extracted_dish,
    }
