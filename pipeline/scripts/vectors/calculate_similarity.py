import typer
from pathlib import Path
import time
import cupy as cp
import json
import time
from wasabi import Printer
from wasabi import table

msg = Printer()


def main(
    condition_path: Path,
    threshold: float,
    batch_size: int,
):
    msg.info(f"Threshold set to {threshold}")

    condition_dict = {}
    with open(condition_path) as reader:
        condition_dict = json.load(reader)
    msg.info(f"Dataset with {len(condition_dict)} conditions loaded")

    condition_dict = calculate_similiarty(condition_dict, threshold, batch_size)

    msg.good("Done")


def calculate_similiarty(dataset: dict, threshold: float, batch: int) -> dict:
    values = list(dataset.values())
    msg.info(f"{len(values)} entities found")
    last_time = time.time()

    key_list = []
    vector_list = []

    for i in range(len(values)):
        current_key = values[i]["key"]
        current_vector = values[i]["vector"]

        vector_list.append(current_vector)
        key_list.append(current_key)

    A = cp.asarray(vector_list)
    norms_reciprocal = 1.0 / cp.linalg.norm(A, axis=-1)

    for i in range(A.shape[0]):
        cosines_i = A.dot(A[i]) * norms_reciprocal * norms_reciprocal[i]

        current_key = key_list[i]
        for k in range(len(cosines_i)):
            if current_key != key_list[k] and cosines_i[k] >= threshold:
                dataset[current_key]["alias"].append(key_list[k])
                dataset[key_list[k]]["alias"].append(current_key)

        if i % batch == 0:
            time_elapsed = round(time.time() - last_time, 2)
            last_time = time.time()
            percentage = round((i / len(values)) * 100, 2)
            time_left = round(((len(values) - i) / batch) * time_elapsed, 2)

            msg.info(
                f"{percentage}% done | {batch} processed in {time_elapsed} | done in {time_left}s"
            )

    return dataset


if __name__ == "__main__":
    typer.run(main)
