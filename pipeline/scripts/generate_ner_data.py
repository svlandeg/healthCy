import pyodbc
import pandas as pd
import json
import random
import typer
from pathlib import Path


def main(output: Path, samples: int):
    conn = pyodbc.connect(
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=localhost;"
        "Database=healthsea;"
        "Trusted_Connection=yes;"
    )
    print("--- Connected to Healthsea SQL ---")

    review_query = pd.read_sql_query("SELECT * FROM healthsea.dbo.clean_reviews", conn)
    review_df = pd.DataFrame(review_query)

    review_sample = review_df.sample(n=samples)
    json_file = []

    for index, row in review_sample.iterrows():
        json_entry = {"text": ""}
        _text = (str(row["title"]) + ". " + str(row["text"])).strip()
        if len(_text) > 512:
            continue
        json_entry["text"] = _text
        json_file.append(json_entry)

    with open(output, "w") as fp:
        json.dump(json_file, fp)


if __name__ == "__main__":
    typer.run(main)
