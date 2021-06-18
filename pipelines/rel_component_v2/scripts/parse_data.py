import spacy
import json
import typer
from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
import random
from wasabi import Printer
from pathlib import Path

from helper_function.functions import get_tokens, calculate_tensor, create_pairs

msg = Printer()

mask_entities = ["CONDITION", "BENEFIT"]
relations = ["RELATED"]

path_to_dep = "../assets/dependencies.csv"
path_to_pos = "../assets/partofspeech.csv"


def main(
    json_loc: Path,
    train_file: Path,
    dev_file: Path,
    eval_split: float,
    use_gpu: bool,
):
    """Creating the corpus from the Prodigy annotations."""
    Doc.set_extension("rel", default={})
    vocab = Vocab()

    if use_gpu:
        spacy.prefer_gpu()

    nlp = spacy.load("en_core_web_lg", exclude=["ner", "lemmatizer"])
    ner_nlp = spacy.load("../../ner_component/training/model-best")

    docs = []
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            if example["answer"] == "accept":

                words = [t["text"] for t in example["tokens"]]
                spaces = [t["ws"] for t in example["tokens"]]
                doc = Doc(vocab, words=words, spaces=spaces)

                tokens = get_tokens(example["text"], nlp, ner_nlp)
                pairs = calculate_tensor(
                    create_pairs(tokens),
                    mask_entities,
                    relations,
                    use_gpu,
                    path_to_dep,
                    path_to_pos,
                )

                for relation in example["relations"]:
                    key1 = (
                        relation["head_span"]["token_start"],
                        relation["head_span"]["token_end"],
                        relation["child_span"]["token_start"],
                        relation["child_span"]["token_end"],
                    )
                    key2 = (
                        relation["child_span"]["token_start"],
                        relation["child_span"]["token_end"],
                        relation["head_span"]["token_start"],
                        relation["head_span"]["token_end"],
                    )

                    if key1 in pairs:
                        pairs[key1]["relation"][relation["label"]] = 1
                    elif key2 in pairs:
                        pairs[key2]["relation"][relation["label"]] = 1

                doc._.rel = pairs
                docs.append(doc)

    split = int(len(docs) * eval_split)
    random.shuffle(docs)

    train = docs[split:]
    dev = docs[:split]

    msg.info(f"{eval_split} training/eval split | {len(docs)} total annotations")

    docbin = DocBin(docs=train, store_user_data=True)
    docbin.to_disk(train_file)
    msg.info(f"{len(train)} training sentences")

    docbin = DocBin(docs=dev, store_user_data=True)
    docbin.to_disk(dev_file)
    msg.info(f"{len(dev)} dev sentences")


if __name__ == "__main__":
    typer.run(main)
