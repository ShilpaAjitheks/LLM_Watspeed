import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from cuisine_classifier.__main__ import load_config, load_dataset, predict_cuisine
from retrieval.vector_store import NOMIC, MPNET

st.set_page_config(page_title="Cuisine Classifier", page_icon="🍽️")

st.title("🍽️ Cuisine Classifier")
st.caption("Enter a dish name to classify its cuisine type.")


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


config, dataset = init()

dish_name = st.text_input("Dish name", placeholder="e.g. Vegetable Quesadillas")
use_dataset = st.checkbox("Look it up in the recipe dataset", value=True)
use_retrieval = st.checkbox("Use retrieval-augmented few-shot", value=False)

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

if st.button("Classify", disabled=not dish_name.strip()):
    with st.spinner("Classifying..."):
        try:
            result, prompt, system_prompt, context_found, few_shot_examples = predict_cuisine(
                dish_name.strip(),
                config,
                dataset=dataset if use_dataset else None,
                use_retrieval=use_retrieval,
                vector_store=vector_store,
            )
        except Exception as e:
            st.error(f"Unexpected error during classification: {e}")
            result, prompt, system_prompt, context_found, few_shot_examples = None, None, None, False, None

    if result:
        # Mode summary
        mode_parts = []
        if use_dataset and context_found:
            mode_parts.append("dataset context")
        elif use_dataset and not context_found:
            mode_parts.append("dish name only (not found in dataset)")
        if use_retrieval:
            mode_parts.append(f"retrieval few-shot ({embed_model})")
        mode_label = " + ".join(mode_parts) if mode_parts else "zero-shot (dish name only)"
        st.info(f"Mode: {mode_label}")

        st.subheader(f"Cuisine: {result['cuisine']}")
        st.progress(result["confidence_score"] / 100, text=f"Confidence: {result['confidence_score']}%")
        for thought in result.get("thoughts", []):
            st.write(f"- {thought}")

        if use_retrieval and few_shot_examples:
            with st.expander("Retrieved few-shot examples"):
                for ex in few_shot_examples:
                    st.markdown(f"**{ex['dish_name']}** → `{ex['cuisine_type']}` (distance: {ex['distance']})")

        with st.expander("Prompt sent to LLM"):
            st.markdown("**System Prompt**")
            st.code(system_prompt, language="text")
            st.markdown("**User Prompt**")
            st.code(prompt, language="text")

    elif result is None and prompt is not None:
        st.error("Classification failed — Ollama did not return a valid response. Make sure Ollama is running.")
