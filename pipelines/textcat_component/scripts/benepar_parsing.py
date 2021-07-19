import benepar, spacy
from spacy.tokens import Span, Doc, Token

from typing import Tuple, List, Iterable, Optional, Dict, Callable, Any

import typer
from pathlib import Path

import json
import secrets


def main(ner_model: Path):

    spacy.prefer_gpu()
    nlp = spacy.load(ner_model)
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    example_batch = [
        "This helped joint pain barely",
        "This helped asthma but made inflammation worse",
        "This is great for energy, muscle growth and bone health",
        "This increases energy and decreases fatigue",
        "This helps hair, nails and skin to grow stronger",
    ]

    for example in example_batch:
        # review = example["text"]
        review = example
        doc = nlp(review)

        statements = extract_clauses(doc)
        print(doc)
        for statement in statements:
            print(f"  >> {statement}")
        print()


def extract_clauses(doc):
    return_list = []
    for sentence in doc.sents:
        for clause in sentence._.constituents:
            if clause._.labels:
                if clause._.labels[0] == "S" and clause._.parent != None:
                    return_list.append((clause._.parent, clause._.parent.ents))
                elif clause._.labels[0] == "S" and clause._.parent == None:
                    return_list.append((clause, clause.ents))
    return return_list


if __name__ == "__main__":
    typer.run(main)
