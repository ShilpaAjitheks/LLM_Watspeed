import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from cuisine_classifier.__main__ import load_config, load_dataset, predict_cuisine

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
def init_vector_store():
    try:
        from retrieval.vector_store import VectorStore
        vs = VectorStore()
        if vs.count() == 0:
            return None, "Vector store is empty — run the population script first."
        return vs, None
    except Exception as e:
        return None, str(e)


config, dataset = init()
vector_store, vs_error = init_vector_store()

dish_name = st.text_input("Dish name", placeholder="e.g. Vegetable Quesadillas")
use_dataset = st.checkbox("Look it up in the recipe dataset", value=True)
use_retrieval = st.checkbox("Use retrieval-augmented few-shot", value=False)

if use_retrieval and vs_error:
    st.warning(f"Retrieval unavailable: {vs_error}")
    use_retrieval = False

if st.button("Classify", disabled=not dish_name.strip()):
    with st.spinner("Classifying..."):
        result = predict_cuisine(
            dish_name.strip(),
            config,
            dataset=dataset if use_dataset else None,
            use_retrieval=use_retrieval,
            vector_store=vector_store,
        )

    if result:
        st.subheader(f"Cuisine: {result['cuisine_type']}")
        st.progress(result["confidence_score"] / 100, text=f"Confidence: {result['confidence_score']}%")
        st.write(result["reasoning"])

        if use_retrieval and vector_store:
            query_text = dish_name.strip()
            examples = vector_store.query(query_text, k=3)
            with st.expander("Retrieved examples"):
                for ex in examples:
                    st.markdown(f"**{ex['dish_name']}** → `{ex['cuisine_type']}` (distance: {ex['distance']})")
    else:
        st.error("Classification failed. Make sure Ollama is running.")
