import spacy
import typer
from pathlib import Path
import pyodbc
import pandas as pd

import json
import time

from wasabi import Printer
import custom_components

msg = Printer()


def main(model: Path, output: Path, gpu: int, batch_size: int):

    # Loading reviews
    conn = pyodbc.connect(
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=localhost;"
        "Database=healthsea;"
        "Trusted_Connection=yes;"
    )
    msg.good("Connected to Healthsea SQL")

    review_query = pd.read_sql_query("SELECT * FROM healthsea.dbo.clean_reviews", conn)
    review_df = pd.DataFrame(review_query)
    total_reviews = len(review_df)
    msg.info(f"{total_reviews} Reviews loaded")

    # Loading dataset
    if not output.exists():
        msg.info("Creating dataset output file")
        dataset = {"processed_ids": [], "reviews": []}
        with open(output, "w", encoding="utf-8") as writer:
            json.dump(dataset, writer)

    dataset = {}
    with open(output) as reader:
        dataset = json.load(reader)
    msg.info(f"Dataset with {len(dataset['reviews'])} reviews loaded")

    # Loading model
    if gpu == 0:
        spacy.prefer_gpu()
        msg.info("Enabled gpu")

    nlp = spacy.load(model)
    msg.info("Healthsea pipeline loaded")

    # Analysis
    msg.divider(f"Starting analysis with batch size {batch_size}")

    counter = 0
    last_time = time.time()
    for index, row in review_df.iterrows():
        review_json = {
            "id": str(row["id"]),
            "p_id": str(row["p_id"]),
            "title": str(row["title"]),
            "text": str(row["text"]),
            "rating": int(row["rating"]),
            "helpful": int(row["helpful"]),
            "customer": str(row["customer"]),
            "date": str(row["date"]),
        }
        review_body = (review_json["title"] + ". " + review_json["text"]).strip()

        if (
            review_json["id"] not in dataset["processed_ids"]
            and len(review_body) < 512
            and len(review_body) > 5
        ):
            try:
                doc = nlp(review_body)
            except AssertionError:
                continue

            review_json["effect"] = doc._.effect_summary

            dataset["processed_ids"].append(review_json["id"])
            dataset["reviews"].append(review_json)
            counter += 1

        if counter % batch_size == 0 and counter != 0:
            with open(output, "w", encoding="utf-8") as writer:
                json.dump(dataset, writer)

            time_elapsed = round(time.time() - last_time, 2)
            last_time = time.time()
            reviews_left = total_reviews - len(dataset["processed_ids"])
            time_left = round((reviews_left / batch_size) * time_elapsed, 2)

            msg.info(
                f"Saved at {len(dataset['processed_ids'])} | {reviews_left} reviews left | Estimated time {time_left}s | {time_elapsed}s per batch"
            )

    msg.divider(f"Analysis done")


if __name__ == "__main__":
    typer.run(main)
