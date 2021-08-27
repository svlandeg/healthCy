import streamlit as st
import pandas as pd
import numpy as np
import json


# Configuration
product_path = "data/healthsea/healthsea_products.json"
substance_path = "data/healthsea/healthsea_substance.json"
condition_path = "data/healthsea/healthsea_condition.json"


@st.cache
def load_data(product_path, substance_path, condition_path):
    products = {}
    with open(product_path) as reader:
        products = json.load(reader)

    substances = {}
    with open(substance_path) as reader:
        substances = json.load(reader)

    conditions = {}
    with open(condition_path) as reader:
        conditions = json.load(reader)

    return products, substances, conditions


st.title("Healthsea 🦑")
st.subheader(
    f"Create better access to health by automatically analyzing reviews to supplementary products"
)
load_state = st.text("Loading data...")
products, substances, conditions = load_data(
    product_path, substance_path, condition_path
)
load_state.text("Loading... done! 🎉")
