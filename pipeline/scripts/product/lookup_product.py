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

    product_query = pd.read_sql_query(
        "SELECT * FROM healthsea.dbo.clean_products", conn
    )
    product_df = pd.DataFrame(product_query)
    total_products = len(product_df)
    msg.info(f"{total_products} Products loaded")

    # Group customer reviews
    msg.divider(f"Start to build product dict")

    dataset = {}
    for index, row in product_df.iterrows():
        product_id = row["id"]

        product_entry = {
            "id": row["id"],
            "brand": row["brand"],
            "name": row["name"],
            "substance": row["substance"],
            "rating": row["rating"],
            "rating_count": row["ratingCount"],
            "review_count": row["reviewCount"],
        }

        dataset[product_id] = product_entry

    with open(output, "w", encoding="utf-8") as writer:
        json.dump(dataset, writer)


if __name__ == "__main__":
    typer.run(main)
