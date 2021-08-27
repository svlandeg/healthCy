from typing import Tuple
import typer
from pathlib import Path
import pandas as pd

import multiprocessing as mp

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
    condition_path: Path,
    benefit_path: Path,
    output_path: Path,
    threshold: float,
    batch_size: int,
):

    cpu_count = 10
    msg.info(f"Cpu count: {cpu_count}")
    msg.info(f"Threshold set to {threshold}")

    reviews = {}
    with open(review_path) as reader:
        reviews = json.load(reader)
    msg.info(f"Dataset with {len(reviews)} reviews loaded")

    condition_dict = {}
    with open(condition_path) as reader:
        condition_dict = json.load(reader)
    msg.info(f"Dataset with {len(condition_dict)} conditions loaded")
    condition_count = len(condition_dict)

    benefit_dict = {}
    with open(benefit_path) as reader:
        benefit_dict = json.load(reader)
    msg.info(f"Dataset with {len(benefit_dict)} benefits loaded")
    benefit_count = len(benefit_dict)

    pool = mp.Pool(cpu_count)
    msg.info(f"Initialized multiprocessing pool")

    condition_dict_batch = divide_dataset(condition_dict, cpu_count)
    benefit_dict_batch = divide_dataset(benefit_dict, cpu_count)
    msg.divider(f"Datasets divided")

    condition_data = []
    for i in condition_dict_batch:
        condition_data.append((condition_dict_batch[i], threshold, batch_size, i))

    benefit_data = []
    for i in benefit_dict_batch:
        benefit_data.append((benefit_dict_batch[i], threshold, batch_size, i))

    condition_dict_results = pool.starmap(cluster_entities, condition_data)
    benefit_dict_results = pool.starmap(cluster_entities, benefit_data)

    condition_dict_2 = {}
    for condition_batch_dict in condition_dict_results:
        for condition in condition_batch_dict:
            condition_dict_2[condition] = condition_batch_dict[condition]

    benefit_dict_2 = {}
    for benefit_batch_dict in benefit_dict_results:
        for benefit in benefit_batch_dict:
            benefit_dict_2[benefit] = benefit_batch_dict[benefit]

    msg.info(f"Clustered vectors")

    clustered_conditions = 0
    clustered_benefits = 0

    msg.info(f"Original Condition dict: {condition_count} -> {len(condition_dict)}")
    msg.info(f"Original Benefit dict: {benefit_count} -> {len(benefit_dict)}")

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


def cluster_entities(dataset, threshold, batch, name):

    values = list(dataset.values())
    msg.info(f"CPU {name}: {len(values)} entities found")
    last_time = time.time()

    cluster_count = 0

    for i in range(len(values)):
        current_key = values[i]["key"]
        current_vector = np.array(values[i]["vector"])
        iterations = 0

        for k in range(i + 1, len(values)):
            second_key = values[k]["key"]
            second_vector = np.array(values[k]["vector"])
            cos_sim = dot(current_vector, second_vector) / (
                norm(current_vector) * norm(second_vector)
            )

            if cos_sim >= threshold:
                cluster_count += 1
                dataset[current_key]["alias"].append(second_key)
                dataset[second_key]["alias"].append(current_key)
            iterations += 1

        if i % batch == 0:
            time_elapsed = round(time.time() - last_time, 2)
            last_time = time.time()
            percentage = round((i / len(values)) * 100, 2)
            time_left = round(((len(values) - i) / batch) * time_elapsed, 2)

            msg.info(
                f"CPU {name}: {percentage}% done | {batch} processed in {time_elapsed} | done in {time_left}s | {iterations} Iterations | {cluster_count*2} entities clustered"
            )

    return dataset


def divide_dataset(dataset, size):
    split_every = int(len(dataset) / size)
    msg.info(
        f"Splitting dataset of {len(dataset)} entries to {size} batches ({split_every})"
    )

    index = 0
    return_dict = {}

    for i in range(size):
        return_dict[i] = {}

    for key in dataset:
        for i in range(size):
            if i == size - 1:
                return_dict[i][key] = dataset[key]

            elif len(return_dict[i]) < split_every:
                return_dict[i][key] = dataset[key]
                break

    size_list = []
    for i in range(size):
        size_list.append(len(return_dict[i]))

    msg.good(f"Divided dataset into: {size_list}")

    return return_dict


if __name__ == "__main__":
    typer.run(main)
