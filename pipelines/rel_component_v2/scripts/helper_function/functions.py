from contextvars import Token
from typing import Dict, List
import cupy
import numpy as np
import spacy
import json

from spacy.tokens import Doc

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
    dep_list: list,
    pos_list: list,
) -> Dict:
    """Calculate tensor from token pairs"""
    if use_gpu:
        import cupy

    pair_dict = {}
    for pair in pairs:

        dep_dict = create_dict(dep_list)
        pos_dict = create_dict(pos_list)

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
        ).astype(np.float64)

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
            "tuple": [pair["tuple"][0]["text"], pair["tuple"][1]["text"]],
            "tensor": input_tensor,
            "relation": {},
        }

        for relation in relations:
            pair_entry["relation"][relation] = 0.0

        pair_dict[pair_key] = pair_entry

    return pair_dict


# Support functions
def create_dict(li: list) -> Dict:
    """transform dictionary to vector"""
    returnDict = {}
    for label in li:
        returnDict[label] = 0
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

    dep_list = None
    pos_list = None

    with open("../../assets/dependencies.json", "r") as f:
        dep_list = json.load(f)

    with open("../../assets/partofspeech.json", "r") as f:
        pos_list = json.load(f)

    use_gpu = True
    if use_gpu:
        spacy.prefer_gpu()

    nlp = spacy.load("../../../ner_component/training/model-best")
    example = "This product helped my joint pain. This has relieved my pain!"
    doc = nlp(example)

    tokens = get_tokens(doc)
    pairs = create_pairs(tokens)
    tensors = calculate_tensor(
        pairs,
        mask_entities,
        relations,
        use_gpu,
        dep_list,
        pos_list,
    )

    print(f"Tokens: {len(tokens)}")
    for token in tokens:
        print(f"{token['text']} | Pos: {token['pos']}")

    print()

    print(f"Pairs: {len(pairs)}")
    for pair in pairs:
        print(f"{pair['text']}")

    print()

    print(f"Tensors: {len(tensors)}")
    for key in tensors:
        print(f"{tensors[key]['tuple']}")
