from typing import Tuple
from numpy.core.fromnumeric import sort
import typer
from pathlib import Path
import pyodbc
import pandas as pd
import spacy

import numpy as np
from numpy import dot
from numpy.linalg import norm

import json
import time

from wasabi import Printer
from wasabi import table

msg = Printer()


def main(
    review_path: Path,
    tok2vec_path: Path,
    output_condition: Path,
    output_benefit: Path,
    output: Path,
    threshold: float,
    gpu: int,
):

    if gpu == 0:
        spacy.prefer_gpu()
    nlp = spacy.load(tok2vec_path)
    msg.info("Model loaded")

    reviews = {}
    with open(review_path) as reader:
        reviews = json.load(reader)
    msg.info(f"Dataset with {len(reviews)} reviews loaded")

    condition_dict = {}
    benefit_dict = {}

    for r_key in reviews:
        review = reviews[r_key]
        effect = review["effect"]

        for c_key in effect:
            if effect[c_key]["label"] == "CONDITION":
                if c_key not in condition_dict:
                    condition_name = str(c_key).replace("_", " ")

                    condition_dict[c_key] = {
                        "name": condition_name,
                        "key": c_key,
                        "frequency": 0,
                        "vector": None,
                        "alias": [],
                    }

                    doc = nlp(condition_name)
                    vector = doc[0].tensor
                    for i in range(1, len(doc)):
                        vector += doc[i].tensor
                    vector = vector / len(doc)
                    condition_dict[c_key]["vector"]

                condition_dict[c_key]["frequency"] += 1

            elif effect[c_key]["label"] == "BENEFIT":
                if c_key not in benefit_dict:
                    benefit_name = str(c_key).replace("_", " ")

                    benefit_dict[c_key] = {
                        "name": benefit_name,
                        "key": c_key,
                        "frequency": 0,
                        "vector": None,
                        "alias": [],
                    }

                    doc = nlp(benefit_name)
                    vector = doc[0].tensor
                    for i in range(1, len(doc)):
                        vector += doc[i].tensor
                    vector = vector / len(doc)
                    benefit_dict[c_key]["vector"]

                benefit_dict[c_key]["frequency"] += 1

    condition_dict = cluster_entities(threshold, condition_dict)
    benefit_dict = cluster_entities(threshold, benefit_dict)

    with open(output_condition, "w", encoding="utf-8") as writer:
        json.dump(condition_dict, writer)

    with open(output_benefit, "w", encoding="utf-8") as writer:
        json.dump(benefit_dict, writer)


def cluster_entities(threshold, dataset):

    values = list(dataset.values())

    for i in range(len(values)):
        current_key = values[i]["key"]
        current_vector = np.array(values[i]["vector"])
        for k in range(i + 1, len(values)):
            second_key = values[k]["key"]
            second_vector = np.array(values[k]["vector"])
            cos_sim = dot(current_vector, second_vector) / (
                norm(current_vector) * norm(second_vector)
            )

            if cos_sim >= threshold:
                dataset[current_key]["alias"].append(second_key)
                dataset[second_key]["alias"].append(current_key)

    return dataset


if __name__ == "__main__":
    typer.run(main)
