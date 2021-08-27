from typing import Tuple
from numpy.core.fromnumeric import sort
import typer
from pathlib import Path
import pyodbc
import pandas as pd

import json
import time

from wasabi import Printer
from wasabi import table

msg = Printer()


def main(customer_grouped: Path, product_path: Path, output: Path):
    # Loading customer dataset
    dataset = {}
    with open(customer_grouped) as reader:
        dataset = json.load(reader)

    products = {}
    with open(product_path) as reader:
        products = json.load(reader)
    msg.info(
        f"Dataset with {len(dataset)} customers and {len(products)} products loaded"
    )

    # Group customer reviews
    msg.divider(f"Starting calculating KPI's")

    # Initialization

    header = ("KPI", "Total", "Average", "Mean", "Min", "Max")

    total = {"review_count": {}}
    total["review_count"]["total"] = 0
    total["review_count"]["total_list"] = []

    total["customer"] = {}

    for customer in dataset:
        # Total reviews
        total["review_count"]["total"] += len(dataset[customer]["reviews"])
        total["review_count"]["total_list"].append(len(dataset[customer]["reviews"]))

        # Customer
        total["customer"][customer] = {}
        total["customer"][customer]["per_date"] = {}

        # Date
        for review in dataset[customer]["reviews"]:
            if review["date"] not in total["customer"][customer]["per_date"]:
                total["customer"][customer]["per_date"][review["date"]] = 0
            total["customer"][customer]["per_date"][review["date"]] += 1

        total["customer"][customer]["per_date_kpi"] = {}
        total["customer"][customer]["per_date_kpi"]["total"] = 0
        total["customer"][customer]["per_date_kpi"]["total_list"] = []
        for date in total["customer"][customer]["per_date"]:
            total["customer"][customer]["per_date_kpi"]["total"] += total["customer"][
                customer
            ]["per_date"][date]
            total["customer"][customer]["per_date_kpi"]["total_list"].append(
                total["customer"][customer]["per_date"][date]
            )

    count_customers(dataset, 5)
    total, review_total_data = review_total(total)
    total, per_date_data = per_date(total)
    data = [review_total_data, per_date_data]
    print(table(data, header=header, divider=True))

    # Trustfactor
    total["trustfactor"] = {}
    total["trustfactor"]["shady_rating"] = 0
    total["trustfactor"]["shady_per_date"] = 0
    total["trustfactor"]["shady_per_date_total"] = 0
    total["trustfactor"]["shady_duplicate"] = 0
    total["trustfactor"]["shady_customer"] = 0
    for customer in dataset:
        dataset[customer]["trustfactor"] = {}
        dataset[customer]["trustfactor"]["shady_rating"] = False
        dataset[customer]["trustfactor"]["shady_per_date"] = False
        dataset[customer]["trustfactor"]["shady_duplicate"] = False

        if "reviews" in dataset[customer]:
            if len(dataset[customer]["reviews"]) > 2:
                rating_sum = 0
                review_texts = []

                for review in dataset[customer]["reviews"]:
                    if review["p_id"] in products:
                        product_entry = products[review["p_id"]]
                    else:
                        continue
                    product_name = product_entry["name"]
                    rating_sum += int(review["rating"])

                    review_text = review["text"].replace(product_name, "").strip()

                    # Shady per review
                    if (
                        review_text in review_texts
                        and not dataset[customer]["trustfactor"]["shady_duplicate"]
                    ):
                        dataset[customer]["trustfactor"]["shady_duplicate"] = True
                        total["trustfactor"]["shady_duplicate"] += 1

                    review_texts.append(review_text)

                avg_rating = rating_sum / len(dataset[customer]["reviews"])

                # Shady rating
                if len(dataset[customer]["reviews"]) > 2 * int(
                    round(total["review_count"]["avg"])
                ) and (avg_rating == 50 or avg_rating == 10):
                    dataset[customer]["trustfactor"]["shady_rating"] = True
                    total["trustfactor"]["shady_rating"] += 1

                # Shady per date
                for date in total["customer"][customer]["per_date"]:
                    if total["customer"][customer]["per_date"][date] > int(
                        round(total["review_count"]["avg"])
                    ):
                        dataset[customer]["trustfactor"]["shady_per_date"] = True
                        total["trustfactor"]["shady_per_date"] += 1
                        total["trustfactor"]["shady_per_date_total"] += total[
                            "customer"
                        ][customer]["per_date"][date]
                        break

        if (
            dataset[customer]["trustfactor"]["shady_rating"]
            or dataset[customer]["trustfactor"]["shady_per_date"]
            or dataset[customer]["trustfactor"]["shady_duplicate"]
        ):
            total["trustfactor"]["shady_customer"] += 1

    print(2 * int(round(total["review_count"]["avg"])))

    msg.info(f"Customers with shady rating: {total['trustfactor']['shady_rating']}")
    msg.info(
        f"Customers with shady reviews per day: {total['trustfactor']['shady_per_date']} with an average of {round(total['trustfactor']['shady_per_date_total']/total['trustfactor']['shady_per_date'],2)}"
    )
    msg.info(
        f"Customers with shady duplicate reviews: {total['trustfactor']['shady_duplicate']}"
    )
    msg.info(
        f"{total['trustfactor']['shady_customer']} shady customers (more than 2 reviews) ({round((total['trustfactor']['shady_customer']/len(dataset))*100,2)}%)"
    )


    with open(output, "w", encoding="utf-8") as writer:
        json.dump(dataset, writer)


def count_customers(dataset, n):

    count_dict = {}
    for i in range(1, n + 1):
        count_dict[i] = 0

    for customer in dataset:
        if len(dataset[customer]["reviews"]) in count_dict:
            count_dict[len(dataset[customer]["reviews"])] += 1

    count_sum = 0

    for i in count_dict:
        msg.info(
            f"{count_dict[i]}/{len(dataset)} customers with {i} review(s) ({round((count_dict[i]/len(dataset))*100,2)}%)"
        )
        count_sum += count_dict[i]

    count_rest = len(dataset) - count_sum

    print()

    msg.info(
        f"{count_sum}/{len(dataset)} customers with 1 to {n} review(s) ({round((count_sum/len(dataset))*100,2)}%)"
    )
    msg.info(
        f"{count_rest}/{len(dataset)} customers with more than {n} review(s) ({round((count_rest/len(dataset))*100,2)}%)"
    )



def review_total(total: dict) -> Tuple:

    # Calculation
    total["review_count"]["min"] = min(total["review_count"]["total_list"])
    total["review_count"]["max"] = max(total["review_count"]["total_list"])
    total["review_count"]["avg"] = round(
        total["review_count"]["total"] / len(total["review_count"]["total_list"]), 2
    )
    total["review_count"]["total_list"] = sort(total["review_count"]["total_list"])
    total["review_count"]["mean"] = total["review_count"]["total_list"][
        int(len(total["review_count"]["total_list"]) / 2)
    ]

    # Table data
    data = (
        "Reviews per customer",
        total["review_count"]["total"],
        total["review_count"]["avg"],
        total["review_count"]["mean"],
        total["review_count"]["min"],
        total["review_count"]["max"],
    )

    return total, data


def per_date(total: dict) -> Tuple:

    total["per_day"] = {}
    total["per_day"]["total"] = 0
    total["per_day"]["total_list"] = []

    for customer in total["customer"]:
        total["per_day"]["total"] += total["customer"][customer]["per_date_kpi"][
            "total"
        ]
        total["per_day"]["total_list"] += total["customer"][customer]["per_date_kpi"][
            "total_list"
        ]

    total["per_day"]["total_list"] = sort(total["per_day"]["total_list"])
    total["per_day"]["mean"] = total["per_day"]["total_list"][
        int(len(total["per_day"]["total_list"]) / 2)
    ]
    total["per_day"]["avg"] = round(
        total["per_day"]["total"] / len(total["per_day"]["total_list"]), 2
    )
    total["per_day"]["min"] = min(total["per_day"]["total_list"])
    total["per_day"]["max"] = max(total["per_day"]["total_list"])

    # Table data
    data = (
        "Reviews per date",
        total["per_day"]["total"],
        total["per_day"]["avg"],
        total["per_day"]["mean"],
        total["per_day"]["min"],
        total["per_day"]["max"],
    )

    return total, data


if __name__ == "__main__":
    typer.run(main)
