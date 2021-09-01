from os import P_OVERLAY
from typing import Tuple
import typer
from pathlib import Path
import json
from wasabi import Printer

msg = Printer()


def main(product_path: Path, substance_path: Path, output_path: Path):
    products = {}
    with open(product_path) as reader:
        products = json.load(reader)
    msg.info(f"Dataset with {len(products)} products loaded")

    substance = {}
    with open(substance_path) as reader:
        substance = json.load(reader)
    msg.info(f"Dataset with {len(substance)} substances loaded")

    healthsea_conditions = {}

    for p_key in products:
        product = products[p_key]

        for c_key in product["effect"]:
            if c_key not in healthsea_conditions:
                healthsea_conditions[c_key] = {
                    "name": product["effect"][c_key]["name"],
                    "label": product["effect"][c_key]["label"],
                    "alias": product["effect"][c_key]["alias"],
                    "products": [],
                    "substance": [],
                }
            healthsea_conditions[c_key]["products"].append(
                (product["effect"][c_key]["relevance_score"], p_key)
            )

    for s_key in substance:
        for c_key in substance[s_key]["effect"]:

            if c_key not in healthsea_conditions:
                healthsea_conditions[c_key] = {
                    "name": product["effect"][c_key]["name"],
                    "label": product["effect"][c_key]["label"],
                    "alias": product["effect"][c_key]["alias"],
                    "products": [],
                    "substance": [],
                }
            healthsea_conditions[c_key]["substance"].append(
                (substance[s_key]["effect"][c_key]["relevance_score"], s_key)
            )

    for c_key in healthsea_conditions:
        healthsea_conditions[c_key]["products"] = sorted(
            healthsea_conditions[c_key]["products"],
            key=lambda tup: tup[0],
            reverse=True,
        )
        healthsea_conditions[c_key]["substance"] = sorted(
            healthsea_conditions[c_key]["substance"],
            key=lambda tup: tup[0],
            reverse=True,
        )

    with open(output_path, "w", encoding="utf-8") as writer:
        json.dump(healthsea_conditions, writer)


if __name__ == "__main__":
    typer.run(main)
