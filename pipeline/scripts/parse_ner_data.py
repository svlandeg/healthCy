import json
import jsonlines
import typer
import random
from wasabi import Printer
from pathlib import Path
import json

msg = Printer()


def main(
    json_loc: Path,
):

    dataset = []

    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            dataset.append(example)

    msg.info(f"Loaded dataset with {len(dataset)} entries")

    conditions = {}
    condition_total = 0
    benefits = {}
    benefit_total = 0
    none_count = 0

    for example in dataset:
        if example["answer"] == "accept":

            if "spans" not in example:
                none_count += 1
                continue

            if len(example["spans"]) == 0:
                none_count += 1
                continue

            for span in example["spans"]:
                token_start = span["token_start"]
                token_end = span["token_end"]
                entitiy = ""
                for token in example["tokens"]:
                    if token["id"] in range(token_start, token_end + 1):
                        entitiy += token["text"]

                entitiy = entitiy.replace(" ", "_").strip().lower()

                if span["label"] == "CONDITION":
                    if entitiy not in conditions:
                        conditions[entitiy] = 0
                    conditions[entitiy] += 1
                    condition_total += 1
                else:
                    if entitiy not in benefits:
                        benefits[entitiy] = 0
                    benefits[entitiy] += 1
                    benefit_total += 1

    msg.info(
        f"Unique conditions: {len(conditions)} Total conditions: {condition_total}"
    )
    msg.info(f"Unique benefits: {len(benefits)} Total benefits: {benefit_total}")
    msg.info(f"Reviews with no entities {none_count}")


if __name__ == "__main__":
    typer.run(main)
