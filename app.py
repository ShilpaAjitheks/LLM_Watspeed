import streamlit as st
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from cuisine_classifier.__main__ import load_config, load_dataset, predict_cuisine
from cuisine_classifier.vision import extract_recipe_raw, parse_extraction, check_extraction
from retrieval.vector_store import NOMIC, MPNET
from retrieval.knowledge_store import get_knowledge_collection, populate_knowledge_store

st.set_page_config(page_title="Cuisine Classifier", page_icon="🍽️")

st.title("🍽️ Cuisine Classifier")
st.caption("Enter a dish name to classify its cuisine type.")

ADAPTER_MODEL = "cuisine-classifier:latest"


@st.cache_resource
def init():
    config = load_config()
    project_root = Path(__file__).parent
    dataset_path = project_root / config["dataset"]["path"]
    dataset = load_dataset(str(dataset_path))
    return config, dataset


@st.cache_resource
def init_vector_store(model):
    try:
        from retrieval.vector_store import VectorStore
        vs = VectorStore(model=model)
        if vs.count() == 0:
            return None, f"Vector store for '{model}' is empty — run: uv run python retrieval/populate.py --model {model}"
        return vs, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def init_knowledge_store():
    try:
        col = get_knowledge_collection()
        if col.count() < 6:
            return False, "Knowledge store not initialised — run: uv run python retrieval/knowledge_store.py"
        return True, None
    except Exception as e:
        return False, str(e)


def adapter_available():
    try:
        import ollama
        models = [m.model for m in ollama.list().models]
        return any(ADAPTER_MODEL in m for m in models)
    except Exception:
        return False


config, dataset = init()

dish_name = st.text_input("Dish name", placeholder="e.g. Vegetable Quesadillas")

uploaded_file = st.file_uploader(
    "Upload a recipe card (optional)", type=["png", "jpg", "jpeg"],
    help="Upload a recipe card image to extract ingredients and description automatically."
)

image_ingredients = None
image_description = None

if uploaded_file is not None:
    file_id = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("extracted_file_id") != file_id:
        st.image(uploaded_file, width=400)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.status("Extracting from image...", expanded=True) as status:
            st.write(f"Calling vision model (`llava:7b`)...")
            raw = extract_recipe_raw(tmp_path)

            st.write("Parsing JSON output...")
            extracted = parse_extraction(raw)

            if extracted is None:
                st.write("⚠️ JSON parse failed — attempting correction with `gemma2:2b`...")

            st.write("Running structural completeness check...")
            quality = check_extraction(extracted, dish_name.strip() if dish_name.strip() else "")

            structural_fields = ("dish_name", "ingredients", "description")
            structural_issues = [i for i in quality["issues"] if any(f in i for f in structural_fields)]
            if not structural_issues and extracted is not None:
                st.write("✅ Structural check: passed")
            else:
                st.write(f"❌ Structural check: {'; '.join(structural_issues)}")

            st.write("Running semantic plausibility check...")
            semantic_issues = [i for i in quality["issues"] if i not in structural_issues]
            if not semantic_issues and quality["passed"]:
                st.write("✅ Semantic check: passed")
            elif semantic_issues:
                st.write(f"❌ Semantic check: {'; '.join(semantic_issues)}")

            if quality["passed"]:
                status.update(label="Extraction complete", state="complete", expanded=False)
            else:
                status.update(label="Extraction failed", state="error", expanded=True)

        st.session_state["extracted_file_id"] = file_id
        st.session_state["extraction_result"] = (extracted, quality)
    else:
        st.image(uploaded_file, width=400)
        extracted, quality = st.session_state["extraction_result"]

    if quality["passed"]:
        st.success("Extracted successfully")
        image_ingredients = ", ".join(extracted["ingredients"])
        image_description = extracted["description"]
    else:
        st.error(f"Quality check failed — {'; '.join(quality['issues'])}.")
        st.warning("This image is not context-friendly. You can upload a different recipe card above, or remove the image to use context enhancement instead.")

model_choice = st.radio(
    "Model",
    ["Base model", "Adapter (Option B)", "Compare both"],
    horizontal=True,
)

if model_choice in ("Adapter (Option B)", "Compare both") and not adapter_available():
    st.warning(
        f"Adapter `{ADAPTER_MODEL}` is not registered in Ollama. "
        f"Run: `ollama create cuisine-classifier -f Modelfile`"
    )

use_dataset = st.checkbox("Look it up in the recipe dataset", value=True)
use_retrieval = st.checkbox("Use retrieval-augmented few-shot", value=False)
use_knowledge_rag = st.checkbox("Use cuisine knowledge RAG", value=False)

if use_retrieval:
    embed_model = st.radio(
        "Embedding model",
        options=[NOMIC, MPNET],
        format_func=lambda m: "nomic-embed-text (Ollama)" if m == NOMIC else "fine-tuned mpnet (local)",
        horizontal=True,
    )
    vector_store, vs_error = init_vector_store(embed_model)
    if vs_error:
        st.warning(f"Retrieval unavailable: {vs_error}")
        use_retrieval = False
else:
    embed_model = None
    vector_store = None

rag_vector_store = None
if use_knowledge_rag:
    ks_ready, ks_error = init_knowledge_store()
    if not ks_ready:
        st.warning(f"Knowledge store: {ks_error}")
        use_knowledge_rag = False
    else:
        rag_vector_store, rag_vs_error = init_vector_store(MPNET)
        if rag_vs_error:
            st.warning(f"Ambiguity signal unavailable (mpnet store empty): {rag_vs_error}")


def run_classify(model_name):
    try:
        return predict_cuisine(
            dish_name.strip(),
            config,
            dataset=dataset if use_dataset else None,
            use_retrieval=use_retrieval,
            vector_store=vector_store,
            model_name=model_name,
            use_rag=use_knowledge_rag,
            rag_vector_store=rag_vector_store,
            image_ingredients=image_ingredients,
            image_description=image_description,
        )
    except Exception as e:
        st.error(f"Error calling {model_name}: {e}")
        return None, None, None, False, None


def show_result(result, prompt, system_prompt, context_found, few_shot_examples, model_name=None):
    if result:
        mode_parts = []
        if image_ingredients is not None:
            mode_parts.append("image extraction")
        elif use_dataset and context_found:
            mode_parts.append("dataset context")
        elif use_dataset and not context_found:
            mode_parts.append("dish name only (not found in dataset)")
        if use_retrieval:
            mode_parts.append(f"retrieval few-shot ({embed_model})")
        if use_knowledge_rag:
            mode_parts.append("cuisine knowledge RAG")
        if model_name and model_name == ADAPTER_MODEL:
            mode_parts.append("LoRA adapter")
        mode_label = " + ".join(mode_parts) if mode_parts else "zero-shot"
        st.info(f"Mode: {mode_label}")

        st.subheader(f"Cuisine: {result['Cuisine']}")
        st.progress(result["Confidence_score"] / 100, text=f"Confidence: {result['Confidence_score']}%")
        for thought in result.get("Reasoning", []):
            st.write(f"- {thought}")

        if result.get("ambiguity"):
            amb = result["ambiguity"]
            level = amb.get("level", "unknown")
            note = amb.get("note", "")
            colour = {"confident": "info", "mild": "warning", "high": "error"}.get(level, "info")
            getattr(st, colour)(f"Ambiguity: **{level}** — {note}")

        if result.get("retrieved_cards"):
            with st.expander("Retrieved cuisine cards"):
                for card in result["retrieved_cards"]:
                    st.markdown(f"**{card['cuisine']}** (distance: {card['distance']})")
                    if card.get("trap"):
                        st.caption(f"⚠️ Trap: {card['trap']}")

        if use_retrieval and few_shot_examples:
            with st.expander("Retrieved few-shot examples"):
                for ex in few_shot_examples:
                    st.markdown(f"**{ex['dish_name']}** → `{ex['cuisine_type']}` (distance: {ex['distance']})")

        with st.expander("Prompt sent to LLM"):
            st.markdown("**System Prompt**")
            st.code(system_prompt, language="text")
            st.markdown("**User Prompt**")
            st.code(prompt, language="text")
    else:
        st.error("Classification failed — check that Ollama is running and the model is available.")


if st.button("Classify", disabled=not dish_name.strip()):
    if model_choice == "Compare both":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Base model** (`{config['model']['name']}`)")
            with st.spinner("Running base model..."):
                show_result(*run_classify(config["model"]["name"]), model_name=config["model"]["name"])
        with col2:
            st.markdown(f"**Adapter (Option B)** (`{ADAPTER_MODEL}`)")
            if adapter_available():
                with st.spinner("Running adapter..."):
                    show_result(*run_classify(ADAPTER_MODEL), model_name=ADAPTER_MODEL)
            else:
                st.warning(f"Adapter `{ADAPTER_MODEL}` not registered in Ollama.")
    else:
        selected_model = config["model"]["name"] if model_choice == "Base model" else ADAPTER_MODEL
        with st.spinner("Classifying..."):
            show_result(*run_classify(selected_model), model_name=selected_model)
