import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
import typer
import numpy as np
from pathlib import Path
from wasabi import Printer
from spacy.scorer import PRFScore

# make the factory work
from rel_pipeline import make_relation_extractor

# make the config work
from rel_model import create_relation_model, create_classification_layer

msg = Printer()


def main(test_file: Path, use_gpu: bool, model: Path):

    if use_gpu:
        spacy.prefer_gpu()

    nlp = spacy.load(model)

    rel_model = None
    threshold = None
    for name, proc in nlp.pipeline:
        if name == "relation_extractor":
            rel_model = proc.get_model()
            threshold = proc.threshold
            break

    doc_bin = DocBin().from_disk(test_file)
    docs = list(doc_bin.get_docs(nlp.vocab))

    msg.good(f"Loaded Model with threshold {threshold}")
    msg.good(f"Loaded {len(docs)} docs for evaluation")
    print("")

    score_dict = {}

    for i in range(len(docs)):
        msg.text(f"------------------------------ \n")
        msg.text(f"{i+1}. {docs[i]}")

        # Create Statement
        doc = nlp(docs[i].text)
        relations = doc._.rel

        for ents in doc.ents:
            ent_key = (ents.start, ents.end - 1)
            entity = ents.text
            label = ents.label_
            t_before = ""
            t_after = ""

            for r_key in relations:
                if relations[r_key]["relation"]["RELATED"] == 1:
                    if ent_key[0] == r_key[0] and ent_key[1] == r_key[1]:
                        t_after += relations[r_key]["tuple"][1] + " "
                    elif ent_key[0] == r_key[2] and ent_key[1] == r_key[3]:
                        t_before += relations[r_key]["tuple"][0] + " "

            classification_string = (f"{t_before}{entity} {t_after}").strip()
            msg.info(f"Predicted statement: {classification_string} | {ent_key}")

        # Error Analysis
        pairs = docs[i]._.rel
        pair_count = len(pairs)
        incorrect_pred = 0
        for pair in pairs:
            expected_outcome = pairs[pair]["relation"]
            input_tensor = rel_model.ops.asarray([pairs[pair]["tensor"]])
            predicted_outcome = rel_model.predict(input_tensor)
            max_val = np.max(predicted_outcome)

            for relation, score in zip(expected_outcome, predicted_outcome):

                if relation not in score_dict:
                    score_dict[relation] = PRFScore()

                threshold_score = 0
                is_wrong = False
                if score >= threshold and score == max_val:
                    threshold_score = 1

                if threshold_score == 1 and expected_outcome[relation] == 1:
                    score_dict[relation].tp += 1
                elif threshold_score == 0 and expected_outcome[relation] == 1:
                    score_dict[relation].fn += 1
                    incorrect_pred += 1
                    is_wrong = True
                elif threshold_score == 1 and expected_outcome[relation] == 0:
                    score_dict[relation].fp += 1
                    incorrect_pred += 1
                    is_wrong = True

                if is_wrong:
                    msg.warn(
                        f"{pairs[pair]['tuple']} | Prediction: {threshold_score} ({float(score):.3f}) Expected: {expected_outcome[relation]}"
                    )
                # else:
                #    msg.good(
                #        f"{pairs[pair]['tuple']} | Prediction: {threshold_score} ({float(score):.3f}) Expected: {expected_outcome[relation]}"
                #    )

        if incorrect_pred == 0:
            msg.good(f" All {pair_count} token pairs correctly predicted! ")
        else:
            msg.warn(
                f"{incorrect_pred} of {pair_count} token pairs were predicted incorrect! "
            )

        msg.text(f"------------------------------ \n")

    for relation in score_dict:
        msg.good(
            f"F-Score {float(score_dict[relation].fscore):.3f} , Recall {float(score_dict[relation].recall):.3f}, Precision {float(score_dict[relation].precision):.3f}"
        )


if __name__ == "__main__":
    typer.run(main)
