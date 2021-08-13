import pyodbc
import pandas as pd
import json
import random
import spacy
from spacy.tokens import Span, Doc, Token

from typing import Tuple, List, Iterable, Optional, Dict, Callable, Any

import typer
from pathlib import Path

import json

from custom_components import extract_clauses


def main(ner_model: Path, output: Path, gpu: int):

    # Connection to database
    conn = pyodbc.connect(
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=localhost;"
        "Database=healthsea;"
        "Trusted_Connection=yes;"
    )
    cursor = conn.cursor()
    print("--- Connected to Healthsea SQL ---")

    # Retrieve review data
    review_data = []
    cursor.execute("SELECT * FROM healthsea.dbo.clean_reviews")
    for row in cursor:
        review_data.append(row)

    review_query = pd.read_sql_query("SELECT * FROM healthsea.dbo.clean_reviews", conn)
    review_df = pd.DataFrame(review_query)

    # Setup spaCy
    if gpu != -1:
        spacy.prefer_gpu()

    nlp = spacy.load(ner_model)
    # nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    filter_keyword = []

    # Generate data
    review_sample = review_df.sample(n=10000)
    json_file = []

    for index, row in review_sample.iterrows():
        text = (str(row["title"]) + ". " + str(row["text"])).strip()
        if len(text) > 512:
            continue

        doc = nlp(text)
        hasKeyword = False

        if len(filter_keyword) > 0:
            for keyword in filter_keyword:
                if keyword in doc.text:
                    hasKeyword = True

        else:
            hasKeyword = True

        if len(doc.ents) > 0 and hasKeyword:
            clauses = extract_clauses(doc)

            for clause in clauses:
                json_entry = {"text": ""}
                json_entry["text"] = str(clause[0])
                json_entry["meta"] = {"entities": str(clause[1])}
                json_file.append(json_entry)

    print(len(json_file))
    with open(output, "w") as fp:
        json.dump(json_file, fp)


if __name__ == "__main__":
    typer.run(main)
