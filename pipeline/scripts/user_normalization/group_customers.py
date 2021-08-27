import typer
from pathlib import Path
import pyodbc
import pandas as pd

import json
import time

from wasabi import Printer

msg = Printer()


def main(output: Path):
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

    # Group customer reviews
    msg.divider(f"Starting grouping reviews to customers")

    dataset = {}
    for index, row in review_df.iterrows():
        customer_id = row["customer"]

        if customer_id not in dataset:
            dataset[customer_id] = {"reviews": [], "trustfactor": {}, "shady": False}

        review_entry = {
            "p_id": row["p_id"],
            "title": row["title"],
            "text": row["text"],
            "rating": row["rating"],
            "helpful": row["helpful"],
            "date": row["date"],
        }
        dataset[customer_id]["reviews"].append(review_entry)

    with open(output, "w", encoding="utf-8") as writer:
        json.dump(dataset, writer)


if __name__ == "__main__":
    typer.run(main)
