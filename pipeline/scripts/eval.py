import spacy
import json
import typer
from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
import random
from wasabi import Printer
from pathlib import Path
import json
import operator

import statement_component


def main(model: Path):

    spacy.prefer_gpu()
    nlp = spacy.load(model)

    text = "This product is really great. I use it everyday to improve my skin and it really supports my arthritis. This made my asthma worse. It makes me sick. I am diagnosed with reflux syndrome."

    doc = nlp(text)
    statements = doc._.statements

    print(doc.text)
    for statement in statements:
        classification = max(statement[2].items(), key=operator.itemgetter(1))[0]
        print(f"   >> {str(statement[1])} ({classification}) '{str(statement[0])}'")


if __name__ == "__main__":
    typer.run(main)
