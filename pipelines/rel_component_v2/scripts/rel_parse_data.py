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
):
    """Creating the corpus from the Prodigy annotations."""
    if use_gpu:
        spacy.prefer_gpu()

    nlp = spacy.load(model)
    relations = ["RELATED"]

    pair_count = 0
    pair_with_relation = 0

    docs = []
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            if example["answer"] == "accept":

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
                    relations,
                )
                pair_count += len(pairs)
                pair_not_found = 0

                # Set unrelated as 1
                # for pair in pairs:
                #    pairs[pair]["relation"]["UNRELATED"] = 1.0

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
                        pair_with_relation += 1
                    elif key2 in pairs:
                        pairs[key2]["relation"][relation["label"]] = 1.0
                        pair_with_relation += 1
                    else:
                        pair_not_found += 1

                if not doc.has_extension("rel"):
                    doc.set_extension("rel", default={})
                doc._.rel = pairs

                docs.append(doc)

    if pair_not_found != 0:
        msg.warn(f"{pair_not_found} pairs not found")

    msg.info(
        f"{pair_with_relation} of {pair_count} pairs have a relation ({float(pair_with_relation/pair_count):0.2f}%)"
    )

    if eval_split != 0:
        split = int(len(docs) * eval_split)
        random.shuffle(docs)

        train = docs[split:]
        dev = docs[:split]

        msg.info(f"{pair_count} total pairs")
        msg.info(f"{eval_split} training/eval split | {len(docs)} total annotations")

        docbin = DocBin(docs=train, store_user_data=True)
        docbin.to_disk(train_file)
        msg.info(f"{len(train)} training sentences")

        docbin = DocBin(docs=dev, store_user_data=True)
        docbin.to_disk(dev_file)
        msg.info(f"{len(dev)} dev sentences")

    else:
        msg.info(f"{pair_count} total pairs | {len(docs)} total annotations")

        docbin = DocBin(docs=docs, store_user_data=True)
        docbin.to_disk(train_file)


if __name__ == "__main__":
    typer.run(main)
