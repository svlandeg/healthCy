import spacy
import json
import typer
from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
import random
from wasabi import Printer
from pathlib import Path
import json

import textcat_pipeline


def main(ner_model: Path, textcat_model: Path):

    spacy.prefer_gpu()
    # ner = spacy.load(ner_model)
    nlp = spacy.load(textcat_model)
    # print(textcat.pipeline)
    # textcat_healthsea = textcat.pipeline[5][1]

    text = "This product is really great. I use it everyday to improve my skin and it really supports my arthritis."

    # doc = ner(text)
    # doc_2 = textcat_healthsea(doc)
    doc = nlp(text)

    statements = doc._.statements

    print(doc)
    print("")
    for statement in statements:
        print(f"  >> {statement} \n")


if __name__ == "__main__":
    typer.run(main)
