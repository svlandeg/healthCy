from typing import Tuple
import typer
from pathlib import Path
import pandas as pd

import multiprocessing as mp

import time

import cupy as cp

import json
import time

from wasabi import Printer
from wasabi import table

msg = Printer()


def main(
    review_path: Path,
    condition_path: Path,
    benefit_path: Path,
    output_path: Path,
    threshold: float,
    batch_size: int,
):
    msg.info(f"Threshold set to {threshold}")

    reviews = {}
    with open(review_path) as reader:
        reviews = json.load(reader)
    msg.info(f"Dataset with {len(reviews)} reviews loaded")

    condition_dict = {}
    with open(condition_path) as reader:
        condition_dict = json.load(reader)
    msg.info(f"Dataset with {len(condition_dict)} conditions loaded")

    benefit_dict = {}
    with open(benefit_path) as reader:
        benefit_dict = json.load(reader)
    msg.info(f"Dataset with {len(benefit_dict)} benefits loaded")

    clustered_conditions = 0
    clustered_benefits = 0
    condition_dict_2 = cluster_entities(condition_dict, threshold, batch_size)
    benefit_dict_2 = cluster_entities(benefit_dict, threshold, batch_size)

    for r_key in reviews:
        for entity in reviews[r_key]["effect"]:
            if reviews[r_key]["effect"][entity]["label"] == "CONDITION":
                reviews[r_key]["effect"][entity]["alias"] = condition_dict_2[entity][
                    "alias"
                ]
                clustered_conditions += len(condition_dict_2[entity]["alias"])
            else:
                reviews[r_key]["effect"][entity]["alias"] = benefit_dict_2[entity][
                    "alias"
                ]
                clustered_benefits += len(benefit_dict_2[entity]["alias"])

    with open(output_path, "w", encoding="utf-8") as writer:
        json.dump(reviews, writer)

    msg.info(
        f"Clustered conditions: {clustered_conditions} | Clustered benefits: {clustered_benefits} | Total: {clustered_conditions+clustered_benefits}"
    )
    msg.info(f"Finished")


def cluster_entities(dataset, threshold, batch):
    values = list(dataset.values())
    msg.info(f"{len(values)} entities found")
    last_time = time.time()

    cluster_count = 0
    key_list = []
    vector_list = []

    for i in range(len(values)):
        current_key = values[i]["key"]
        current_vector = values[i]["vector"]

        vector_list.append(current_vector)
        key_list.append(current_key)

    A = cp.asarray(vector_list, dtype="float32")
    norms_reciprocal = 1.0 / cp.linalg.norm(A, axis=-1)

    for i in range(A.shape[0]):
        cosines_i = A.dot(A[i]) * norms_reciprocal * norms_reciprocal[i]
        reached_threshold = cp.asnumpy(cosines_i >= threshold)

        current_key = key_list[i]
        for k in range(i + 1, len(reached_threshold)):
            if reached_threshold[k]:
                cluster_count += 2
                dataset[current_key]["alias"].append(key_list[k])
                dataset[key_list[k]]["alias"].append(current_key)

        if i % batch == 0:
            time_elapsed = round(time.time() - last_time, 2)
            last_time = time.time()
            percentage = round((i / len(values)) * 100, 2)
            time_left = round(((len(values) - i) / batch) * time_elapsed, 2)

            msg.info(
                f"{percentage}% done | {batch} processed in {time_elapsed} | done in {time_left}s | {cluster_count} entities clustered"
            )

    return dataset


if __name__ == "__main__":
    typer.run(main)
