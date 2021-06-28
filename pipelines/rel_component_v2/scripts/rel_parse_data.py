import spacy
import json
import typer
from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
import random
from wasabi import Printer
from pathlib import Path

from rel_pipeline import get_tokens, calculate_tensor, create_pairs

msg = Printer()


def main(
    json_loc: Path,
    train_file: Path,
    dev_file: Path,
    eval_split: float,
    use_gpu: bool,
    model: Path,
    dep: Path,
    pos: Path,
):
    """Creating the corpus from the Prodigy annotations."""
    # Doc.set_extension("rel", default={})
    vocab = Vocab()

    if use_gpu:
        spacy.prefer_gpu()

    nlp = spacy.load(model)
    mask_entities = ["CONDITION", "BENEFIT"]
    relations = ["RELATED"]

    dep_list = None
    pos_list = None

    with open(dep, "r") as f:
        dep_list = json.load(f)

    with open(pos, "r") as f:
        pos_list = json.load(f)

    docs = []
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            if example["answer"] == "accept":

                # words = [t["text"] for t in example["tokens"]]
                # spaces = [t["ws"] for t in example["tokens"]]
                # blank_doc = Doc(vocab, words=words, spaces=spaces)

                doc = nlp(example["text"])
                ents_list = []

                for span in example["spans"]:
                    ents_list.append(
                        Span(
                            doc,
                            span["token_start"],
                            span["token_end"] + 1,
                            span["label"],
                        )
                    )

                doc.set_ents(ents_list)
                tokens = get_tokens(doc)
                pairs = calculate_tensor(
                    create_pairs(tokens),
                    mask_entities,
                    relations,
                    dep_list,
                    pos_list,
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
                        pairs[key1]["relation"][relation["label"]] = 1.0
                    elif key2 in pairs:
                        pairs[key2]["relation"][relation["label"]] = 1.0

                if not doc.has_extension("rel"):
                    doc.set_extension("rel", default={})
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
