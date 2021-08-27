from typing import Tuple
from numpy.core.fromnumeric import sort
import typer
from pathlib import Path
import pandas as pd
import spacy

import time

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
    condition_output: Path,
    benefit_output: Path,
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
                    condition_dict[c_key]["vector"] = vector.tolist()

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
                    benefit_dict[c_key]["vector"] = vector.tolist()

                benefit_dict[c_key]["frequency"] += 1

    with open(condition_output, "w", encoding="utf-8") as writer:
        json.dump(condition_dict, writer)

    with open(benefit_output, "w", encoding="utf-8") as writer:
        json.dump(benefit_dict, writer)

    msg.info(f"Calculated and saved vectors")


if __name__ == "__main__":
    typer.run(main)
