import spacy
import json
import typer
from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
import random
from wasabi import Printer
from wasabi import table

from pathlib import Path
import json
import operator
from spacy.scorer import PRFScore

import custom_components

from wasabi import Printer

msg = Printer()


def main(model: Path, eval: Path, ner_annotations: Path, textcat_annotations: Path):

    spacy.prefer_gpu()
    nlp = spacy.load(model)

    evaluation = None
    with open(eval, "r", encoding="utf8") as jsonfile:
        evaluation = json.load(jsonfile)

    ner_annotations_data = []
    with open(ner_annotations, "r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            ner_annotations_data.append(json.loads(line))

    textcat_annotations_data = []
    with open(textcat_annotations, "r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            textcat_annotations_data.append(json.loads(line))

    # print(evaluation)

    scorer = {
        "CONDITION": PRFScore(),
        "BENEFIT": PRFScore(),
        "POSITIVE": PRFScore(),
        "NEGATIVE": PRFScore(),
        "NEUTRAL": PRFScore(),
    }
    data = {
        "CONDITION": [],
        "BENEFIT": [],
        "POSITIVE": 0,
        "NEGATIVE": 0,
        "NEUTRAL": 0,
    }
    uncorrect_ner = []
    uncorrect_textcat = []

    for eval in evaluation:

        for entity_index in eval["effect_summary"]:
            if entity_index not in data[eval["effect_summary"][entity_index]["label"]]:
                data[eval["effect_summary"][entity_index]["label"]].append(entity_index)
            data[eval["effect_summary"][entity_index]["classification"]] += 1

        text = eval["text"]
        doc = nlp(text)
        effect_summary_gold = eval["effect_summary"]
        effect_summary_pred = doc._.effect_summary

        entities_gold = list(effect_summary_gold.keys())
        entities_pred = list(effect_summary_pred.keys())

        middle_join = [value for value in entities_gold if value in entities_pred]
        left_join = [value for value in entities_gold if value not in entities_pred]
        right_join = [value for value in entities_pred if value not in entities_gold]

        for entity in middle_join:
            if (
                effect_summary_pred[entity]["label"]
                == effect_summary_gold[entity]["label"]
            ):
                scorer[effect_summary_gold[entity]["label"]].tp += 1
            else:
                scorer[effect_summary_gold[entity]["label"]].fn += 1
                scorer[effect_summary_pred[entity]["label"]].fp += 1
                uncorrect_ner.append(
                    f"({eval['number']}) {entity} (Prediction: {effect_summary_pred[entity]['label']}) (Expected: {effect_summary_gold[entity]['label']})"
                )

        for entity in left_join:
            scorer[effect_summary_gold[entity]["label"]].fn += 1
            uncorrect_ner.append(f"({eval['number']}) False negative {entity}")

        for entity in right_join:
            scorer[effect_summary_pred[entity]["label"]].fp += 1
            uncorrect_ner.append(f"({eval['number']}) False positive {entity}")

        for entity_index in middle_join:
            if (
                effect_summary_pred[entity_index]["classification"]
                == effect_summary_gold[entity_index]["classification"]
            ):
                scorer[effect_summary_gold[entity_index]["classification"]].tp += 1
            else:
                uncorrect_textcat.append(
                    f"({eval['number']}) {entity_index} (Prediction: {effect_summary_pred[entity_index]['classification']}) (Expected: {effect_summary_gold[entity_index]['classification']}) ({text})"
                )
                scorer[effect_summary_pred[entity_index]["classification"]].fp += 1
                scorer[effect_summary_gold[entity_index]["classification"]].fn += 1

    msg.divider(f"Total reviews: {len(evaluation)}")
    msg.divider(
        f"Evaluating Named Entity Recognition trained with {len(ner_annotations_data)} annotations"
    )
    msg.info(
        f"Total examples ({len(data['BENEFIT'])+len(data['CONDITION'])}), CONDITION ({len(data['CONDITION'])}), BENEFIT ({len(data['BENEFIT'])})"
    )

    ner_data = [
        (
            "CONDITION",
            round(scorer["CONDITION"].fscore, 2),
            round(scorer["CONDITION"].recall, 2),
            round(scorer["CONDITION"].precision, 2),
        ),
        (
            "BENEFIT",
            round(scorer["BENEFIT"].fscore, 2),
            round(scorer["BENEFIT"].recall, 2),
            round(scorer["BENEFIT"].precision, 2),
        ),
        (
            "AVERAGE",
            round((scorer["BENEFIT"].fscore + scorer["CONDITION"].fscore) / 2, 2),
            round((scorer["BENEFIT"].recall + scorer["CONDITION"].recall) / 2, 2),
            round((scorer["BENEFIT"].precision + scorer["CONDITION"].precision) / 2, 2),
        ),
    ]
    header = ("Label", "F-Score", "Recall", "Precision")
    widths = (10, 10, 10, 10)
    aligns = ("l", "c", "c", "c")
    print(table(ner_data, header=header, divider=True, widths=widths, aligns=aligns))

    msg.fail(f"{len(uncorrect_ner)} false predictions")
    for error_ner in uncorrect_ner:
        print("  >> " + error_ner)

    msg.divider(
        f"Evaluating Text Classification trained with {len(textcat_annotations_data)} annotations"
    )

    msg.info(
        f"Total examples ({data['POSITIVE']+data['NEGATIVE']+data['NEUTRAL']}), POSITIVE ({data['POSITIVE']}), NEGATIVE ({data['NEGATIVE']}), NEUTRAL ({data['NEUTRAL']})"
    )

    textcat_data = [
        (
            "POSITIVE",
            round(scorer["POSITIVE"].fscore, 2),
            round(scorer["POSITIVE"].recall, 2),
            round(scorer["POSITIVE"].precision, 2),
        ),
        (
            "NEGATIVE",
            round(scorer["NEGATIVE"].fscore, 2),
            round(scorer["NEGATIVE"].recall, 2),
            round(scorer["NEGATIVE"].precision, 2),
        ),
        (
            "NEUTRAL",
            round(scorer["NEUTRAL"].fscore, 2),
            round(scorer["NEUTRAL"].recall, 2),
            round(scorer["NEUTRAL"].precision, 2),
        ),
        (
            "AVERAGE",
            round(
                (
                    scorer["NEUTRAL"].fscore
                    + scorer["NEGATIVE"].fscore
                    + scorer["POSITIVE"].precision
                )
                / 3,
                2,
            ),
            round(
                (
                    scorer["NEUTRAL"].recall
                    + scorer["NEGATIVE"].recall
                    + scorer["POSITIVE"].precision
                )
                / 3,
                2,
            ),
            round(
                (
                    scorer["NEUTRAL"].precision
                    + scorer["NEGATIVE"].precision
                    + scorer["POSITIVE"].precision
                )
                / 3,
                2,
            ),
        ),
    ]
    header = ("Label", "F-Score", "Recall", "Precision")
    widths = (10, 10, 10, 10)
    aligns = ("l", "c", "c", "c")
    print(
        table(textcat_data, header=header, divider=True, widths=widths, aligns=aligns)
    )

    msg.fail(f"{len(uncorrect_textcat)} false predictions")
    for error_textcat in uncorrect_textcat:
        print("  >> " + error_textcat)


def test(model: Path):
    spacy.prefer_gpu()
    nlp = spacy.load(model)

    text = "I have <CONDITION> and this really helped me to breath better"

    doc = nlp(text)

    effects = doc._.effect_summary
    print(doc.cats)
    print(effects)
    print(doc._.statements)


if __name__ == "__main__":
    typer.run(main)
