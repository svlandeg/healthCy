from numpy.core.fromnumeric import product
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import spacy
from pathlib import Path
import operator
from spacy_streamlit import visualize_ner
from annotated_text import annotated_text
import numpy as np
from numpy import dot
from numpy.linalg import norm

import custom_components

# Configuration
product_path = "data/healthsea/healthsea_products.json"
substance_path = "data/healthsea/healthsea_substance.json"
condition_path = "data/healthsea/healthsea_condition.json"
condition_vectors_path = "data/pretrain/condition_vectors.json"
benefit_vectors_path = "data/pretrain/benefit_vectors.json"
model_path = "training/healthsea/config_trf/"


@st.cache(allow_output_mutation=True)
def load_data(
    product_path,
    substance_path,
    condition_path,
    condition_vectors_path,
    benefit_vectors_path,
):
    products = {}
    with open(product_path) as reader:
        products = json.load(reader)

    substances = {}
    with open(substance_path) as reader:
        substances = json.load(reader)

    conditions = {}
    with open(condition_path) as reader:
        conditions = json.load(reader)

    condition_vectors = {}
    with open(condition_vectors_path) as reader:
        condition_vectors = json.load(reader)

    benefit_vectors = {}
    with open(benefit_vectors_path) as reader:
        benefit_vectors = json.load(reader)

    vectors = {}
    for k in condition_vectors:
        vectors[k] = condition_vectors[k]
    for k in benefit_vectors:
        vectors[k] = benefit_vectors[k]

    return products, substances, conditions, vectors


def load_model(path: Path):
    return spacy.load(path)


def kpi(n, text):
    html = f"""
    <div class='KPI'>
        <h1>{n}</h1>
        <span>{text}</span>
    </div>
    """
    return html


def tsne_plot(dataset):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for i in dataset:
        tokens.append(i[0])
        labels.append(i[1])

    tsne_model = TSNE(
        perplexity=40, n_components=3, init="pca", n_iter=2500, random_state=23
    )

    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    z = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        z.append(value[2])

    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        text=labels,
        textposition="top right",
        mode="lines+markers+text",
        marker={
            "size": 10,
            "opacity": 0.8,
        },
    )

    # Configure the layout.
    layout = go.Layout(
        margin={"l": 0, "r": 0, "b": 0, "t": 0}, font={"color": "#DF55E2"}
    )

    data = [trace]

    return go.Figure(data=data, layout=layout)


with open("scripts/presentation/style.css") as f:
    st.markdown("<style>" + f.read() + "</style>", unsafe_allow_html=True)

st.title("Welcome to Healthsea 🪐")
st.subheader(
    f"Create easier access to health by providing easily digestable information ✨"
)

intro, jellyfish = st.columns(2)

jellyfish.image("scripts/presentation/img/jellyfish.png", use_column_width="auto")
intro.markdown(
    "Healthsea automatically analyzes user-generated content about complementary medicine and extracts health effects on certain conditions"
)
intro.markdown(
    "Use this app to discover an interesting usecase in the healthcare domain. Find the most suitable products and substances for every health preferences, based on natural language."
)


products, substances, conditions, vectors = load_data(
    product_path,
    substance_path,
    condition_path,
    condition_vectors_path,
    benefit_vectors_path,
)

st.title("See the healthsea pipeline in action 🔥")
st.subheader(
    f"Write down an example review which describes an effect on a health condition and let the magic happen!"
)

review_example = st.text_input(
    label="Write a review", value="This is great for joint pain!"
)
nlp = load_model(model_path)
doc = nlp(review_example)
statements = doc._.statements
effects = doc._.effect_summary

visualize_ner(
    doc,
    labels=nlp.get_pipe("ner").labels,
    show_table=False,
    title="Detect health conditions",
    colors={"CONDITION": "#FF4B76", "BENEFIT": "#629B68"},
)

color_scheme = {
    "POSITIVE": "#629B68",
    "NEGATIVE": "#FF4B76",
    "NEUTRAL": "#929292",
    "ANAMNESIS": "#FF5959",
}

st.header("Detect health effects")

counter = 0
for statement in statements:
    if statement[1] == None:
        continue
    entity_name = statement[1].replace("_", " ")
    entity_key = statement[1]
    classification = max(statement[2].items(), key=operator.itemgetter(1))[0]
    percentage = round(float(statement[2][classification]) * 100, 2)
    annotated_text(
        (
            f"{str(statement[0])}",
            f"{classification} effect on {entity_name} ({percentage}%)",
            color_scheme[classification],
        )
    )

    if classification == "ANAMNESIS":
        calculated_classification = effects[entity_key]["classification"]
        annotated_text(
            (
                f"{str(statement[0])}",
                f"{calculated_classification} effect on {entity_name} (100%)",
                color_scheme[calculated_classification],
            )
        )
    st.text("")

st.title("Let's look at the analyzed dataset 💥")
st.subheader(
    f"Search for a health condition and find the most suitable products and substances 💊"
)

review_count = 0
for p_key in products:
    review_count += len(products[p_key]["reviews"])

condition_count = 0
benefit_count = 0
for c_key in conditions:
    if conditions[c_key]["label"] == "CONDITION":
        condition_count += 1
    else:
        benefit_count += 1


kpi1, kpi2, kpi3 = st.columns(3)
kpi1.markdown(kpi(review_count, "Reviews"), unsafe_allow_html=True)
kpi2.markdown(kpi(len(products), "Products"), unsafe_allow_html=True)
kpi3.markdown(kpi(len(substances), "Substances"), unsafe_allow_html=True)
kpi4, kpi5 = st.columns(2)
kpi4.markdown(kpi(condition_count, "Conditions"), unsafe_allow_html=True)
kpi5.markdown(kpi(benefit_count, "Benefits"), unsafe_allow_html=True)

# Condition search

health_condition = st.text_input(
    label="Search for a health condition", value="joint pain"
)
top_n = st.slider(
    label="Show top n entries",
    min_value=10,
    max_value=100,
    step=1,
)

condition_key = str(health_condition).replace(" ", "_").lower().strip()

if condition_key not in conditions:
    doc2 = nlp(str(health_condition))
    health_vector = doc2[0].tensor
    for i in range(1, len(doc2)):
        health_vector += doc2[i].tensor
    health_vector = health_vector / len(doc2)

    similarity_list = []
    for c_key in vectors:
        other_vector = np.array(vectors[c_key]["vector"])
        cos_sim = dot(health_vector, other_vector) / (
            norm(health_vector) * norm(other_vector)
        )
        similarity_list.append((c_key, cos_sim))
    condition_key = max(similarity_list, key=operator.itemgetter(1))[0]


alias_dataset = []
alias_dataset.append(
    (np.array(vectors[condition_key]["vector"]), vectors[condition_key]["name"])
)
for alias in conditions[condition_key]["alias"]:
    alias_dataset.append((np.array(vectors[alias]["vector"]), vectors[alias]["name"]))

if len(alias_dataset) > 2:
    st.write(tsne_plot(alias_dataset))

st.header(f"Products with highest score for {health_condition}")

product_score_list = []

if len(conditions[condition_key]["products"]) >= top_n:
    range_to_n = top_n
else:
    range_to_n = len(conditions[condition_key]["products"])

for i in range(range_to_n):
    product_score_list.append(
        (
            conditions[condition_key]["products"][i][0],
            conditions[condition_key]["products"][i][1],
            conditions[condition_key]["name"],
        )
    )

for alias in conditions[condition_key]["alias"]:
    if len(conditions[alias]["products"]) >= top_n:
        range_to_n = top_n
    else:
        range_to_n = len(conditions[alias]["products"])
    for i in range(range_to_n):
        product_score_list.append(
            (
                conditions[alias]["products"][i][0],
                conditions[alias]["products"][i][1],
                conditions[alias]["name"],
            )
        )

product_score_list = sorted(product_score_list, key=lambda tup: tup[0], reverse=True)[
    :top_n
]

product_data = {"p_id": [], "product": [], "health_condition": [], "score": []}
for score in product_score_list:
    product_data["p_id"].append(score[1])
    product_data["product"].append(products[score[1]]["name"])
    product_data["health_condition"].append(score[2])
    product_data["score"].append(score[0])

product_df = pd.DataFrame(data=product_data)

st.write(product_df)

st.header(f"Substances with highest score for {health_condition}")

substance_score_list = []

if len(conditions[condition_key]["substance"]) >= top_n:
    range_to_n = top_n
else:
    range_to_n = len(conditions[condition_key]["substance"])

for i in range(range_to_n):
    substance_score_list.append(
        (
            conditions[condition_key]["substance"][i][0],
            conditions[condition_key]["substance"][i][1],
            conditions[condition_key]["name"],
        )
    )

for alias in conditions[condition_key]["alias"]:
    if len(conditions[alias]["substance"]) >= top_n:
        range_to_n = top_n
    else:
        range_to_n = len(conditions[alias]["substance"])
    for i in range(range_to_n):
        substance_score_list.append(
            (
                conditions[alias]["substance"][i][0],
                conditions[alias]["substance"][i][1],
                conditions[alias]["name"],
            )
        )

substance_score_list = sorted(
    substance_score_list, key=lambda tup: tup[0], reverse=True
)[:top_n]

substance_data = {"substance": [], "health_condition": [], "score": []}
for score in substance_score_list:
    substance_data["substance"].append(score[1])
    substance_data["health_condition"].append(score[2])
    substance_data["score"].append(score[0])

substance_df = pd.DataFrame(data=substance_data)


st.write(substance_df)

st.write(conditions[condition_key])
