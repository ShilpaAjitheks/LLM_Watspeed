import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cuisine_classifier.__main__ import load_config, load_dataset, predict_cuisine

st.set_page_config(page_title="Cuisine Classifier", page_icon="🍽️")

st.title(" Cuisine Classifier")
st.caption("Enter a dish name to classify its cuisine type.")


@st.cache_resource
def init():
    config = load_config()
    import os
    package_dir = Path(__file__).parent / "src" / "cuisine_classifier"
    project_root = Path(__file__).parent
    dataset_path = project_root / config["dataset"]["path"]
    dataset = load_dataset(str(dataset_path))
    return config, dataset


config, dataset = init()

dish_name = st.text_input("Dish name", placeholder="e.g. Vegetable Quesadillas")
use_dataset = st.checkbox("Look it up in the recipe dataset", value=True)

if st.button("Classify", disabled=not dish_name.strip()):
    with st.spinner("Classifying..."):
        result = predict_cuisine(dish_name.strip(), config, dataset=dataset if use_dataset else None)

    if result:
        st.subheader(f"Cuisine: {result['cuisine_type']}")
        st.progress(result["confidence_score"] / 100, text=f"Confidence: {result['confidence_score']}%")
        st.write(result["reasoning"])
    else:
        st.error("Classification failed. Make sure Ollama is running.")
