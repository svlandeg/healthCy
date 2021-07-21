import spacy
import json
import typer
from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
import random
from wasabi import Printer
from pathlib import Path
import json

from extract_clauses import extract_clauses

msg = Printer()


def main(ner: Path, textcat: Path):

    textcat = spacy.load(textcat)
    ner = spacy.load(ner)

    example_batch = [
        "This helped joint pain barely",
        "This helped asthma but made inflammation worse",
        "This is great for energy, muscle growth and bone health",
        "This increases energy and decreases fatigue",
        "This helps hair, nails and skin to grow stronger",
    ]

    for example in example_batch:
        review = example
        doc = ner(review)

        statements = extract_clauses(doc)
        print(doc)
        for statement in statements:
            classification_doc = textcat(statement[0])
            classification = classification_doc.cats
            print(f"  >> {statement} | {classification}")
        print()


if __name__ == "__main__":
    typer.run(main)
