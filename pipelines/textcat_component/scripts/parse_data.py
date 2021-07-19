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

    vocab = Vocab()

    examples = []

    label_dict = {}
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            examples.append(example)

            labels = example["accept"]

            for label in labels:
                if label not in label_dict:
                    label_dict[label] = []

    for example in examples:
        if example["answer"] == "accept":

            # Parse the tokens
            words = [t["text"] for t in example["tokens"]]
            spaces = [t["ws"] for t in example["tokens"]]
            doc = Doc(vocab, words=words, spaces=spaces)

            labels = example["accept"]

            cats_dict = {}
            for label in labels:
                for label2x in label_dict.keys():
                    cats_dict[label2x] = 0.0

                cats_dict[label] = 1.0
                doc._.cats = cats_dict
                label_dict[label].append(doc)

    train = []
    dev = []

    for label in label_dict:
        split = int(len(label_dict[label]) * eval_split)
        train += label_dict[label][split:]
        dev += label_dict[label][:split]

        msg.info(
            f"{label}: {len(label_dict[label])} ({len(label_dict[label])/len(example):.2f}%) | Train: {len(label_dict[label][split:])} Dev: {len(label_dict[label][:split])})"
        )

    docbin = DocBin(docs=train, store_user_data=True)
    docbin.to_disk(train_file)
    msg.info(f"{len(train)} training sentences")

    docbin = DocBin(docs=dev, store_user_data=True)
    docbin.to_disk(dev_file)
    msg.info(f"{len(dev)} dev sentences")


if __name__ == "__main__":
    typer.run(main)
