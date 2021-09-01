import typer
from pathlib import Path

import json

from wasabi import Printer
from wasabi import table

msg = Printer()


def main(dataset_path: Path, customer_trustscore: Path, output: Path):

    dataset = {}
    with open(dataset_path) as reader:
        dataset = json.load(reader)
    msg.info(f"Dataset with {len(dataset['reviews'])} reviews loaded")

    customers = {}
    with open(customer_trustscore) as reader:
        customers = json.load(reader)
    msg.info(f"Dataset with {len(customers)} customers loaded")

    review_scored = {}

    for review in dataset["reviews"]:

        if len(review["effect"]) == 0:
            continue

        review["trustscore"] = customers[review["customer"]]["trustfactor"]

        if (
            review["trustscore"]["shady_rating"]
            or review["trustscore"]["shady_per_date"]
            or review["trustscore"]["shady_duplicate"]
        ):
            customer_score = 0.25
        else:
            customer_score = 1

        rating = int(review["rating"])
        helpful_score = 1 + (int(review["helpful"]) / 10)

        if rating == 10:
            rating_score = -1
        elif rating >= 20 and rating < 30:
            rating_score = -0.5
        elif rating >= 40 and rating < 50:
            rating_score = 0.5
        elif rating == 50:
            rating_score = 1
        else:
            rating_score = 0

        for condition in review["effect"]:

            if review["effect"][condition]["classification"] == "POSITIVE":
                base_score = 1
            elif review["effect"][condition]["classification"] == "NEGATIVE":
                base_score = -1
            else:
                base_score = 0.25

            review["effect"][condition]["score"] = (
                (base_score + rating_score) * helpful_score
            ) * customer_score

        review_scored[review["id"]] = review

    with open(output, "w", encoding="utf-8") as writer:
        json.dump(review_scored, writer)

    calculate_overview(dataset)


def calculate_overview(dataset):
    condition_dict = {}
    has_condition = 0

    total_count = {
        "CONDITION": 0,
        "BENEFIT": 0,
        "POSITIVE": 0,
        "NEUTRAL": 0,
        "NEGATIVE": 0,
    }

    for review in dataset["reviews"]:
        if len(review["effect"]) > 0:
            has_condition += 1
            for condition in review["effect"]:
                if condition not in condition_dict:
                    condition_dict[condition] = {
                        "POSITIVE": 0,
                        "NEUTRAL": 0,
                        "NEGATIVE": 0,
                        "label": review["effect"][condition]["label"],
                    }
                condition_dict[condition][
                    review["effect"][condition]["classification"]
                ] += 1
                total_count[review["effect"][condition]["classification"]] += 1
                total_count[review["effect"][condition]["label"]] += 1

    total_mention = (
        total_count["POSITIVE"] + total_count["NEUTRAL"] + total_count["NEGATIVE"]
    )

    msg.info(
        f"{has_condition} out of {len(dataset['reviews'])} reviews include health entities ({round((has_condition/len(dataset['reviews']))*100,2)}%)"
    )

    msg.divider("Overview KPI's")

    table_data = [
        ("CONDITION", total_count["CONDITION"]),
        ("BENEFIT", total_count["BENEFIT"]),
        ("POSITIVE", total_count["POSITIVE"]),
        ("NEUTRAL", total_count["NEUTRAL"]),
        ("NEGATIVE", total_count["NEGATIVE"]),
        ("TOTAL MENTION", total_mention),
    ]
    header = ("LABEL", "COUNT")
    widths = (
        15,
        15,
    )
    aligns = ("c", "c")
    print(table(table_data, header=header, divider=True, widths=widths, aligns=aligns))

    condition_ranking = []
    benefit_ranking = []
    for condition in condition_dict:
        total = (
            condition_dict[condition]["POSITIVE"]
            + condition_dict[condition]["NEUTRAL"]
            + condition_dict[condition]["NEGATIVE"]
        )
        if condition_dict[condition]["label"] == "CONDITION":
            condition_ranking.append(
                (
                    condition,
                    total,
                    condition_dict[condition]["POSITIVE"],
                    condition_dict[condition]["NEUTRAL"],
                    condition_dict[condition]["NEGATIVE"],
                )
            )
        else:
            benefit_ranking.append(
                (
                    condition,
                    total,
                    condition_dict[condition]["POSITIVE"],
                    condition_dict[condition]["NEUTRAL"],
                    condition_dict[condition]["NEGATIVE"],
                )
            )

    condition_ranking_total = sorted(
        condition_ranking, key=lambda x: x[1], reverse=True
    )
    condition_ranking_positive = sorted(
        condition_ranking, key=lambda x: x[2], reverse=True
    )
    condition_ranking_negative = sorted(
        condition_ranking, key=lambda x: x[4], reverse=True
    )

    benefit_ranking_total = sorted(benefit_ranking, key=lambda x: x[1], reverse=True)
    benefit_ranking_positive = sorted(benefit_ranking, key=lambda x: x[2], reverse=True)
    benefit_ranking_negative = sorted(benefit_ranking, key=lambda x: x[4], reverse=True)

    msg.divider("Top 10 mentioned conditions")
    condition_ranking_total_data = condition_ranking_total[:10]
    condition_ranking_total_header = (
        "CONDITION",
        "COUNT",
        "POSITIVE",
        "NEUTRAL",
        "NEGATIVE",
    )
    condition_ranking_total_widths = (20, 10, 10, 10, 10)
    condition_ranking_total_aligns = ("l", "c", "c", "c", "c")
    print(
        table(
            condition_ranking_total_data,
            header=condition_ranking_total_header,
            divider=True,
            widths=condition_ranking_total_widths,
            aligns=condition_ranking_total_aligns,
        )
    )

    msg.divider("Top 10 improved conditions")
    condition_ranking_positive_data = condition_ranking_positive[:10]
    print(
        table(
            condition_ranking_positive_data,
            header=condition_ranking_total_header,
            divider=True,
            widths=condition_ranking_total_widths,
            aligns=condition_ranking_total_aligns,
        )
    )
    msg.divider("Top 10 worsen conditions")
    condition_ranking_negaitve_data = condition_ranking_negative[:10]
    print(
        table(
            condition_ranking_negaitve_data,
            header=condition_ranking_total_header,
            divider=True,
            widths=condition_ranking_total_widths,
            aligns=condition_ranking_total_aligns,
        )
    )

    msg.divider("Top 10 mentioned benefits")
    benefit_ranking_total_data = benefit_ranking_total[:10]
    benefit_ranking_total_header = (
        "BENEFIT",
        "COUNT",
        "POSITIVE",
        "NEUTRAL",
        "NEGATIVE",
    )
    print(
        table(
            benefit_ranking_total_data,
            header=benefit_ranking_total_header,
            divider=True,
            widths=condition_ranking_total_widths,
            aligns=condition_ranking_total_aligns,
        )
    )

    msg.divider("Top 10 improved benefits")
    benefit_ranking_positive_data = benefit_ranking_positive[:10]
    print(
        table(
            benefit_ranking_positive_data,
            header=benefit_ranking_total_header,
            divider=True,
            widths=condition_ranking_total_widths,
            aligns=condition_ranking_total_aligns,
        )
    )

    msg.divider("Top 10 worsen benefits")
    benefit_ranking_negative_data = benefit_ranking_negative[:10]
    print(
        table(
            benefit_ranking_negative_data,
            header=benefit_ranking_total_header,
            divider=True,
            widths=condition_ranking_total_widths,
            aligns=condition_ranking_total_aligns,
        )
    )


if __name__ == "__main__":
    typer.run(main)
