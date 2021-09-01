from typing import Tuple
import typer
from pathlib import Path
import json
from wasabi import Printer
import re

msg = Printer()


def main(
    review_path: Path,
    product_path: Path,
    substance_path: Path,
    product_output_path: Path,
    substance_output_path: Path,
):

    reviews = {}
    with open(review_path) as reader:
        reviews = json.load(reader)
    msg.info(f"Dataset with {len(reviews)} reviews loaded")

    products = {}
    with open(product_path) as reader:
        products = json.load(reader)
    msg.info(f"Dataset with {len(products)} products loaded")

    substance = {}
    with open(substance_path) as reader:
        substance = json.load(reader)
    msg.info(f"Dataset with {len(substance)} products+substance loaded")

    healthsea_products = {}

    product_without_substance = 0
    for p_key in products:
        if p_key not in healthsea_products:
            healthsea_products[p_key] = products[p_key]
            healthsea_products[p_key]["reviews"] = {}
            healthsea_products[p_key]["substance"] = {}
            if p_key in substance:
                if substance[p_key] != "None":
                    healthsea_products[p_key]["substance"] = substance[p_key]

    review_without_product = 0
    for r_key in reviews:
        p_id = reviews[r_key]["p_id"]

        if p_id in healthsea_products:
            healthsea_products[p_id]["reviews"][r_key] = reviews[r_key]
        else:
            review_without_product += 1

    msg.info(
        f"Succesful grouped | Products without substance: {product_without_substance} | Reviews without products {review_without_product}"
    )

    for p_key in healthsea_products:
        product = healthsea_products[p_key]
        product["effect"] = {}

        for r_key in product["reviews"]:
            review = product["reviews"][r_key]

            for c_key in review["effect"]:
                if c_key not in product["effect"]:
                    name = str(c_key).replace("_", " ")
                    product["effect"][c_key] = {
                        "name": name,
                        "label": review["effect"][c_key]["label"],
                        "score": 0,
                        "occurence": 0,
                        "occurence_alias": 0,
                        "alias": review["effect"][c_key]["alias"],
                    }
                product["effect"][c_key]["score"] += float(
                    review["effect"][c_key]["score"]
                )
                product["effect"][c_key]["occurence"] += 1

    found_alias = 0
    for p_key in healthsea_products:
        product = healthsea_products[p_key]
        total_reviews = int(product["review_count"])

        if total_reviews < len(product["reviews"]):
            msg.fail(f"Product has more reviews than expected")

        for c_key in product["effect"]:
            current_condition = c_key
            for c_key2 in product["effect"]:
                if current_condition in product["effect"][c_key2]["alias"]:
                    product["effect"][c_key]["occurence_alias"] += product["effect"][
                        c_key2
                    ]["occurence"]
                    found_alias += 1

        for c_key in product["effect"]:
            score_multiplier = (
                (
                    product["effect"][c_key]["occurence"]
                    + product["effect"][c_key]["occurence_alias"]
                )
                / total_reviews
            ) + 1
            product["effect"][c_key]["relevance_score"] = (
                product["effect"][c_key]["score"] * score_multiplier
            )

    msg.info(f"Finished scoring | Found alias: {found_alias}")

    healthsea_substance = {}

    for p_key in healthsea_products:
        product = healthsea_products[p_key]

        if product["substance"] != "None":
            for s_key in product["substance"]:
                if s_key not in healthsea_substance:
                    healthsea_substance[s_key] = {
                        "alias": [],
                        "forms": [],
                        "products": [],
                        "effect": {},
                    }

                healthsea_substance[s_key]["alias"].append(
                    product["substance"][s_key]["alias"]
                )
                healthsea_substance[s_key]["forms"] += product["substance"][s_key][
                    "forms"
                ]
                healthsea_substance[s_key]["products"].append(p_key)
                for c_key in product["effect"]:
                    if c_key not in healthsea_substance[s_key]["effect"]:
                        healthsea_substance[s_key]["effect"][c_key] = {
                            "name": product["effect"][c_key]["name"],
                            "alias": product["effect"][c_key]["alias"],
                            "label": product["effect"][c_key]["label"],
                            "relevance_score": 0,
                        }
                    healthsea_substance[s_key]["effect"][c_key][
                        "relevance_score"
                    ] += product["effect"][c_key]["relevance_score"]

    with open(product_output_path, "w", encoding="utf-8") as writer:
        json.dump(healthsea_products, writer)

    with open(substance_output_path, "w", encoding="utf-8") as writer:
        json.dump(healthsea_substance, writer)


if __name__ == "__main__":
    typer.run(main)
