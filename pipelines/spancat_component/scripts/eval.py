import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
import typer
import numpy as np
from pathlib import Path
from wasabi import Printer

msg = Printer()


def main(model: Path, use_gpu: bool):

    if use_gpu:
        spacy.prefer_gpu()

    nlp = spacy.load(model)

    examples = [
        "This helped my joint pain",
        "I got rid of my nasty acne",
        "This made my sore throat worse",
        "This caused constipation",
        "This helped my asthma barely",
    ]
    for example in examples:
        doc = nlp(example)
        print(doc.spans)


if __name__ == "__main__":
    typer.run(main)
