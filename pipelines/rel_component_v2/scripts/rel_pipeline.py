from itertools import islice
from typing import Tuple, List, Iterable, Optional, Dict, Callable, Any

from thinc.types import Floats2d, Floats3d
import thinc.util
import numpy as np
from spacy.training.example import Example
from thinc.api import Model, Optimizer
from spacy.tokens.doc import Doc
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.vocab import Vocab
from spacy import Language
from thinc.model import set_dropout_rate
from wasabi import Printer
from pathlib import Path

import json

Doc.set_extension("rel", default={}, force=True)
msg = Printer()


@Language.factory(
    "relation_extractor",
    requires=["doc.ents", "token.pos", "token.dep"],
    assigns=["doc._.rel"],
    default_score_weights={
        "rel_micro_p": None,
        "rel_micro_r": None,
        "rel_micro_f": None,
    },
)
def make_relation_extractor(
    nlp: Language,
    name: str,
    model: Model,
    *,
    threshold: float,
    batch_size: int,
    dep: Path,
    pos: Path,
):
    """Construct a RelationExtractor component."""
    return RelationExtractor(
        nlp.vocab,
        model,
        name,
        threshold=threshold,
        batch_size=batch_size,
        pos=pos,
        dep=dep,
    )


class RelationExtractor(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "rel",
        *,
        threshold: float,
        batch_size: int,
        dep: Path,
        pos: Path,
    ) -> None:
        """Initialize a relation extractor."""
        self.vocab = vocab
        self.model = model
        self.name = name
        self.cfg = {"labels": [], "threshold": threshold}

        self.batch_size = batch_size

        self.dep_list = None
        self.pos_list = None

        with open(dep, "r") as f:
            self.dep_list = json.load(f)

        with open(pos, "r") as f:
            self.pos_list = json.load(f)

    @property
    def labels(self) -> Tuple[str]:
        """Returns the labels currently added to the component."""
        return tuple(self.cfg["labels"])

    @property
    def threshold(self) -> float:
        """Returns the threshold above which a prediction is seen as 'True'."""
        return self.cfg["threshold"]

    def add_label(self, label: str) -> int:
        """Add a new label to the pipe."""
        if not isinstance(label, str):
            raise ValueError(
                "Only strings can be added as labels to the RelationExtractor"
            )
        if label in self.labels:
            return 0
        self.cfg["labels"] = list(self.labels) + [label]
        return 1

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipe to a Doc."""
        predictions = self.predict([doc])
        self.set_annotations([doc], predictions)
        return doc

    def predict(self, docs: Iterable[Doc]) -> Floats3d:
        """Apply the pipeline's model to a batch of docs, without modifying them."""
        scores = []

        for doc in docs:
            score = []
            entity_list = []
            for ent in doc.ents:
                if ent.label_ not in entity_list:
                    entity_list.append(ent.label_)

            pairs = None
            if not doc.has_extension("rel"):
                tokens = get_tokens(doc)
                pairs = calculate_tensor(
                    create_pairs(tokens),
                    entity_list,
                    self.cfg["labels"],
                    self.dep_list,
                    self.pos_list,
                )
            elif doc.has_extension("rel") and len(doc._.rel) == 0:
                tokens = get_tokens(doc)
                pairs = calculate_tensor(
                    create_pairs(tokens),
                    entity_list,
                    self.cfg["labels"],
                    self.dep_list,
                    self.pos_list,
                )
            else:
                pairs = doc._.rel

            for pair in pairs:
                input_tensor = np.array(pairs[pair]["tensor"]).astype(np.float32)
                input_tensor = self.model.ops.asarray([input_tensor])
                # input_tensor = self.model.ops.asarray(pairs[pair]["tensor"])
                output_tensor = self.model.predict(input_tensor)
                score.append(self.model.ops.asarray(output_tensor))
            scores.append(self.model.ops.asarray(score))

        return scores

    def set_annotations(self, docs: Iterable[Doc], scores: Floats3d) -> None:
        """Modify a batch of `Doc` objects, using pre-computed scores."""
        for doc, score in zip(docs, scores):
            pairs = None

            if not doc.has_extension("rel"):
                tokens = get_tokens(doc)
                pairs = calculate_tensor(
                    create_pairs(tokens),
                    [],
                    self.cfg["labels"],
                    self.dep_list,
                    self.pos_list,
                )
            elif doc.has_extension("rel") and len(doc._.rel) == 0:
                tokens = get_tokens(doc)
                pairs = calculate_tensor(
                    create_pairs(tokens),
                    [],
                    self.cfg["labels"],
                    self.dep_list,
                    self.pos_list,
                )
            else:
                pairs = doc._.rel

            for pair, prediction in zip(pairs, score):
                for pred, relation_key in zip(prediction, pairs[pair]["relation"]):
                    if pred >= self.threshold:
                        pairs[pair]["relation"][relation_key] = float(1.0)
                    else:
                        pairs[pair]["relation"][relation_key] = float(0.0)

            doc._.rel = pairs

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.2,
        set_annotations: bool = False,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to predict and get_loss."""
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        set_dropout_rate(self.model, drop)

        # run the model
        docs = [eg.reference for eg in examples]

        train_x = []
        train_y = []

        for doc in docs:
            for pairs in doc._.rel:
                train_x.append(doc._.rel[pairs]["tensor"])
                train_y.append(np.array(list(doc._.rel[pairs]["relation"].values())))

        train_x = np.array(train_x).astype(np.float32)
        train_y = np.array(train_y).astype(np.float32)
        train_x = self.model.ops.asarray(train_x)
        train_y = self.model.ops.asarray(train_y)

        batches = self.model.ops.multibatch(
            self.batch_size, train_x, train_y, shuffle=True
        )

        for X, Y in batches:
            predictions, backprop = self.model.begin_update(X)
            gradient = predictions - Y
            loss = (gradient ** 2).sum(axis=1).mean()

            backprop(gradient)
            if sgd is not None:
                self.model.finish_update(sgd)
            losses[self.name] += float(loss)

        return losses

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Language = None,
        labels: Optional[List[str]] = None,
    ):
        """Initialize the pipe for training, using a representative set
        of data examples.
        """
        if labels is not None:
            for label in labels:
                self.add_label(label)
        else:
            for example in get_examples():
                pairs = example.reference._.rel
                for pair in pairs:
                    for label in pairs[pair]["relation"]:
                        self.add_label(label)

        subbatch = list(islice(get_examples(), 10))
        doc_sample = [eg.reference for eg in subbatch]

        train_x = []
        train_y = []

        for doc in doc_sample:
            for pair in doc._.rel:
                train_x.append(doc._.rel[pair]["tensor"])
                train_y.append(np.array(list(doc._.rel[pair]["relation"].values())))

        train_x = np.array(train_x).astype(np.float32)
        train_y = np.array(train_y).astype(np.float32)
        train_x = self.model.ops.asarray(train_x)
        train_y = self.model.ops.asarray(train_y)

        self.model.initialize(X=train_x, Y=train_y)
        nI = self.model.get_dim("nI")
        nO = self.model.get_dim("nO")
        print(
            f"Initialized model with input dimension nI={nI} and output dimension nO={nO}"
        )

    def score(self, examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
        """Score a batch of examples."""
        return score_relations(examples, self.threshold)


def score_relations(examples: Iterable[Example], threshold: float) -> Dict[str, Any]:
    """Score a batch of examples."""
    # micro_prf = PRFScore()

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for example in examples:
        gold = example.reference._.rel
        pred = example.predicted._.rel

        # Only for one class
        for pairs_gold, pairs_pred in zip(gold, pred):
            for relation_gold, relation_pred in zip(
                gold[pairs_gold]["relation"], pred[pairs_pred]["relation"]
            ):
                y = gold[pairs_gold]["relation"][relation_gold]
                x = pred[pairs_pred]["relation"][relation_pred]

                if x >= threshold:
                    x = 1
                else:
                    x = 0

                if x == 1 and y == 1:
                    true_positive += 1
                elif x == 0 and y == 1:
                    false_negative += 1
                elif x == 1 and y == 0:
                    false_positive += 1

    precision = 0
    if (true_positive + false_positive) != 0:
        precision = true_positive / (true_positive + false_positive)

    recall = 0
    if (true_positive + false_negative) != 0:
        recall = true_positive / (true_positive + false_negative)

    f_score = 0
    if (precision + recall) != 0:
        f_score = (2 * precision * recall) / (precision + recall)

    return {
        "rel_micro_p": float(precision),
        "rel_micro_r": float(recall),
        "rel_micro_f": float(f_score),
    }


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

                    pos_list = token["pos"]
                    for pos2 in token2["pos"]:
                        if pos2 not in pos_list:
                            pos_list.append(pos2)

                    entry = {
                        "tuple": (token, token2),
                        "text": (token["text"], token2["text"]),
                        "dist": token2["start"] - token["end"],
                        "pos": pos_list,
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
    dep_list: list,
    pos_list: list,
) -> Dict:
    """Calculate tensor from token pairs"""

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
            # Masking
            if token["label"] not in mask_entites:
                for tensor in token["tensor"]:

                    if thinc.util.has_cupy:
                        import cupy

                        if isinstance(tensor, cupy._core.core.ndarray):
                            sum_vector += cupy.asnumpy(tensor)
                        elif isinstance(tensor, np.ndarray):
                            sum_vector += np.asarray(list(tensor))
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
def calculate_dep_dist(token1, token2) -> List[str]:
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
    import spacy
    from spacy.lang.en import English
    from rel_model import create_relation_model, create_classification_layer
    from spacy.tokens import Doc
    from thinc.api import prefer_gpu
    import numpy as np
    from reader import create_docbin_reader
    from thinc.api import Adam
    from spacy.training import Example

    spacy.prefer_gpu()
    prefer_gpu()

    lang = English()

    train_file = "../data/train.spacy"
    docbin_reader = create_docbin_reader(train_file)
    dev_file = "../data/dev.spacy"
    docbin_reader_dev = create_docbin_reader(dev_file)

    model = create_relation_model(create_classification_layer(None, None))
    rel_pipe = make_relation_extractor(lang, "rel_extractor", model, threshold=0.4)
    rel_pipe.initialize(docbin_reader, nlp=lang)

    optimizer = Adam(0.001)
    epochs = 50

    for example in docbin_reader_dev(lang):
        doc = example.reference
        print(doc.has_extension("rel"))

    for i in range(epochs):
        loss = rel_pipe.update(docbin_reader(lang), sgd=optimizer)

        example_list = []
        for example in docbin_reader_dev(lang):
            doc = example.predicted
            prediction = rel_pipe(doc)
            example_list.append(Example(prediction, example.reference))

        score = rel_pipe.score(example_list)
        print(
            f"{i+1}: Loss {loss['rel_extractor']} | F-Score: {float(score['rel_micro_f']):.2f}, Precision: {float(score['rel_micro_p']):.2f}, Recall: {float(score['rel_micro_r']):.2f}"
        )

    nlp = spacy.load("../../ner_component/training/model-best")
    text = "This helped my joint pain"
    test_doc = nlp(text)

    # print(rel_pipe.labels)

    # rel_doc = rel_pipe(test_doc)
    # for pair in rel_doc._.rel:
    #     text = rel_doc._.rel[pair]["tuple"]
    #     predictions = rel_doc._.rel[pair]["relation"]
    #     print(f"{text} | {predictions}")
