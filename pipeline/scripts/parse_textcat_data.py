import spacy
import json
import typer
from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
import random
from wasabi import Printer
from pathlib import Path
import json

msg = Printer()


def main(
    json_loc: Path,
    train_file: Path,
    dev_file: Path,
    eval_split: float,
):
    """Creating the corpus from the Prodigy annotations."""

    examples = []
    label_dict = {}
    nlp = spacy.load(
        "en_core_web_lg", disable=["tagger", "parser", "ner", "lemmatizer"]
    )

    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            examples.append(example)

            labels = example["accept"]

            for label in labels:
                if label not in label_dict:
                    label_dict[label] = []

    msg.info(f"{len(examples)} examples loaded")

    for example in examples:
        if example["answer"] == "accept":

            # Parse the tokens
            doc = nlp(example["text"])

            labels = example["accept"]

            cats_dict = {}
            for label in label_dict.keys():
                cats_dict[label] = 0.0

            for label in labels:
                cats_dict[label] = 1.0

            doc.cats = cats_dict
            label_dict[label].append(doc)

    train = []
    dev = []

    for label in label_dict:
        split = int(len(label_dict[label]) * eval_split)
        train += label_dict[label][split:]
        dev += label_dict[label][:split]

        msg.info(
            f"{label}: {len(label_dict[label])} ({(len(label_dict[label])/len(examples))*100:.2f}%) | Train: {len(label_dict[label][split:])} Dev: {len(label_dict[label][:split])})"
        )

    docbin = DocBin(docs=train, store_user_data=True)
    docbin.to_disk(train_file)
    msg.info(f"{len(train)} training sentences")

    docbin = DocBin(docs=dev, store_user_data=True)
    docbin.to_disk(dev_file)
    msg.info(f"{len(dev)} dev sentences")


if __name__ == "__main__":
    typer.run(main)
