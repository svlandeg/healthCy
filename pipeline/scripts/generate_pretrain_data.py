import pyodbc
import pandas as pd
import json
import random
import typer
from pathlib import Path


def main(output: Path):
    conn = pyodbc.connect(
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=localhost;"
        "Database=healthsea;"
        "Trusted_Connection=yes;"
    )
    print("--- Connected to Healthsea SQL ---")

    review_query = pd.read_sql_query("SELECT * FROM healthsea.dbo.clean_reviews", conn)
    review_df = pd.DataFrame(review_query)

    review_sample = review_df.sample(n=200)
    json_file = ""

    for index, row in review_df.iterrows():
        json_entry = {"text": ""}
        _text = (str(row["title"]) + ". " + str(row["text"])).strip()
        if len(_text) > 512:
            continue
        json_entry["text"] = _text
        json_entry["text"] = json_entry["text"].replace('"', " ").replace("'", " ")
        json_line = (
            str(json_entry)
            .replace("'", '"')
            .replace("\n", " ")
            .replace("\xa0", " ")
            .replace("\\", " ")
        )
        json_file += json_line + "\n"

    with open(output, "w", encoding="utf8") as fp:
        fp.write(json_file)


if __name__ == "__main__":
    typer.run(main)
