from contextvars import Token
from typing import Dict, List
import cupy
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import spacy

from spacy.tokens import Doc, Span

# Main functions
def get_tokens(doc: Doc) -> List[Dict]:
    """Extract token information from doc and merge entities to one"""
    returnList = []

    for sent in doc.sents:
        for tok in sent:

            if tok.is_punct:
                continue

            token_text = tok.text
            token_list = [tok]
            token_doc = sent
            token_tensor = [tok.tensor]
            label = "None"
            pos_tags = [tok.pos_]
            start_token = tok.i
            end_token = start_token

            # Find & Merge entities to one entry
            if doc[tok.i].ent_iob_ == "B":
                label = doc[tok.i].ent_type_

                if start_token + 1 < len(doc):
                    for tok2 in doc[start_token + 1 :]:
                        if (
                            doc[tok2.i].ent_type_ == label
                            and doc[tok2.i].ent_iob_ == "I"
                        ):
                            token_list.append(tok2)
                            pos_tags.append(tok2.pos_)
                            token_tensor.append(tok2.tensor)
                            token_text += f" {tok2.text}"
                            end_token = tok2.i
                        if doc[tok2.i].ent_iob_ == "B":
                            break
            elif doc[tok.i].ent_iob_ == "I":
                continue

            returnList.append(
                {
                    "text": token_text,
                    "tokens": token_list,
                    "label": label,
                    "start": start_token,
                    "end": end_token,
                    "sent": token_doc,
                    "pos": pos_tags,
                    "tensor": token_tensor,
                }
            )
    return returnList


def create_pairs(token_list: List[Dict]) -> List[Dict]:
    """Create token pairs and get their dependencies"""
    pair_list = []
    index = 0
    for token in token_list:
        if index + 1 < len(token_list):
            for token2 in token_list[index + 1 :]:
                if token["sent"] == token2["sent"]:

                    # Get dependencies
                    dep_list = []
                    for tok in token["tokens"]:
                        for tok2 in token2["tokens"]:
                            tmp_list = calculate_dep_dist(tok, tok2)
                            for dep in tmp_list:
                                if dep not in dep_list:
                                    dep_list.append(dep)

                    entry = {
                        "tuple": (token, token2),
                        "text": (token["text"], token2["text"]),
                        "dist": token2["start"] - token["start"],
                        "pos": token["pos"] + token2["pos"],
                        "dep": dep_list,
                        "dep_dist": len(dep_list),
                    }
                    pair_list.append(entry)
        index += 1
    return pair_list


def calculate_tensor(
    pairs: List[Dict],
    mask_entites: List[str],
    relations: List[str],
    use_gpu: bool,
    dep_path: str,
    pos_path: str,
) -> Dict:
    """Calculate tensor from token pairs"""
    if use_gpu:
        import cupy

    pair_dict = {}
    for pair in pairs:
        dep_labels = pd.read_csv(dep_path)
        pos_labels = pd.read_csv(pos_path)

        dep_dict = create_dep_dict(dep_labels)
        pos_dict = create_pos_dict(pos_labels)

        for pos in pair["pos"]:
            if pos.upper() in pos_dict:
                pos_dict[pos.upper()] = 1
            else:
                print(f"{pos} not in pos_dict!")
        for dep in pair["dep"]:
            if dep.upper() in dep_dict:
                dep_dict[dep.upper()] = 1
            else:
                print(f"{dep} not in dep_dict!")

        dep_vector = dict_to_vector(dep_dict)
        pos_vector = dict_to_vector(pos_dict)
        dist_vector = np.array([pair["dist"], pair["dep_dist"]])
        token_vector = None

        sum_vector = np.zeros(len(pair["tuple"][0]["tensor"][0]))
        sum_tokens = 0

        for token in pair["tuple"]:
            if token["label"] not in mask_entites:
                for tensor in token["tensor"]:
                    if use_gpu:
                        sum_vector += cupy.asnumpy(tensor)
                    else:
                        sum_vector += np.asarray(list(tensor))
                    sum_tokens += 1

        if sum_tokens != 0:
            token_vector = sum_vector / sum_tokens
        else:
            continue

        input_tensor = np.concatenate(
            (token_vector, dep_vector, pos_vector, dist_vector)
        )

        pair["relation"] = {}
        for relation in relations:
            pair["relation"][relation] = 0

        pair_key = (
            pair["tuple"][0]["start"],
            pair["tuple"][0]["end"],
            pair["tuple"][1]["start"],
            pair["tuple"][1]["end"],
        )

        pair_entry = {
            "tuple": [pair["tuple"][0]["text"], pair["tuple"][0]["text"]],
            "tensor": input_tensor,
            "relation": {},
        }

        for relation in relations:
            pair_entry["relation"][relation] = 0

        pair_dict[pair_key] = pair_entry

    return pair_dict


# Support functions
def create_dep_dict(df: DataFrame) -> Dict:
    """transform dictionary to vector"""
    returnDict = {}
    for index, row in df.iterrows():
        returnDict[row["Label"]] = 0
    return returnDict


def create_pos_dict(df: DataFrame) -> Dict:
    """transform dictionary to vector"""
    returnDict = {}
    for index, row in df.iterrows():
        returnDict[row["Label"]] = 0
    return returnDict


def dict_to_vector(d: Dict):
    """transform dictionary to vector"""
    return_vector = np.zeros(len(d))
    index = 0
    for key in d:
        return_vector[index] = d[key]
        index += 1
    return return_vector


# Calculate dependency relation
def calculate_dep_dist(token1: Token, token2: Token) -> List[str]:
    """Get dependencies between token1 and token2"""
    ancestors = list(token1.ancestors)
    ancestors2 = list(token2.ancestors)

    dep_list = []

    if token2 in ancestors:
        dep_list.append(token1.dep_)
        for ancestor in ancestors:
            dep_list.append(ancestor.dep_)
            if ancestor == token2:
                break

    elif token1 in ancestors2:
        dep_list.append(token2.dep_)
        for ancestor in ancestors2:
            dep_list.append(ancestor.dep_)
            if ancestor == token1:
                break

    else:
        common_ancestor = None
        for ancestor in ancestors2:
            if ancestor in ancestors:
                common_ancestor = ancestor
                break

        dep_list.append(token1.dep_)
        dep_list.append(token2.dep_)

        for ancestor in ancestors:
            dep_list.append(ancestor.dep_)
            if ancestor == common_ancestor:
                break

        for ancestor in ancestors2:
            if ancestor == common_ancestor:
                break
            elif ancestor.text not in dep_list:
                dep_list.append(ancestor.dep_)

    return dep_list


if __name__ == "__main__":

    mask_entities = ["CONDITION", "BENEFIT"]
    relations = ["RELATED"]
    use_gpu = True

    if use_gpu:
        spacy.prefer_gpu()

    nlp = spacy.load("en_core_web_lg", exclude=["ner", "lemmatizer"])
    ner_nlp = spacy.load("../../../ner_component/training/model-best")

    path_to_dep = "../../assets/dependencies.csv"
    path_to_pos = "../../assets/partofspeech.csv"

    example = "This product helped my joint pain. This has relieved my pain!"

    doc = nlp(example)
    ents = ner_nlp(example)
    doc.ents = ents.ents

    tokens = get_tokens(doc)
    pairs = calculate_tensor(
        create_pairs(tokens),
        mask_entities,
        relations,
        use_gpu,
        path_to_dep,
        path_to_pos,
    )

    for pair_key in pairs:
        print(f" {pair_key}  | {len(pairs[pair_key]['relation'])}")

    # example = "This product helped my joint pain"
    # doc = nlp(example)
    # print(calculate_dep_dist(doc[3], doc[4]))
