from itertools import islice
from typing import Tuple, List, Iterable, Optional, Dict, Callable, Any

from spacy.scorer import PRFScore
from thinc.types import Floats2d, Floats3d
import numpy
from spacy.training.example import Example
from thinc.api import Model, Optimizer
from spacy.tokens.doc import Doc
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.vocab import Vocab
from spacy import Language
from thinc.model import set_dropout_rate
from wasabi import Printer

import json


from helper_function.functions import get_tokens, calculate_tensor, create_pairs

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
    nlp: Language, name: str, model: Model, *, threshold: float
):
    """Construct a RelationExtractor component."""
    return RelationExtractor(nlp.vocab, model, name, threshold=threshold)


class RelationExtractor(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "rel",
        *,
        threshold: float,
    ) -> None:
        """Initialize a relation extractor."""
        self.vocab = vocab
        self.model = model
        self.name = name
        self.cfg = {"labels": [], "threshold": threshold}

        self.dep_list = None
        self.pos_list = None

        with open("../assets/dependencies.json", "r") as f:
            self.dep_list = json.load(f)

        with open("../assets/partofspeech.json", "r") as f:
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

            pairs = None
            if not doc.has_extension("rel"):
                entity_list = []
                for ent in doc.ents:
                    if ent.label_ not in entity_list:
                        entity_list.append(ent.label_)

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
                input_tensor = model.ops.asarray(pairs[pair]["tensor"])
                score.append(self.model.predict(input_tensor))
            scores.append(score)

        return self.model.ops.asarray(scores)

    def set_annotations(self, docs: Iterable[Doc], scores: Floats2d) -> None:
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
            else:
                pairs = doc._.rel

            for pair, prediction in zip(pairs, score):
                for pred, relation_key in zip(prediction, pairs[pair]["relation"]):
                    if pred >= self.threshold:
                        pairs[pair]["relation"][relation_key] = 1.0
                    else:
                        pairs[pair]["relation"][relation_key] = 0.0

            doc._.rel = pairs

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.2,
        set_annotations: bool = False,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
        batch_size: int = 128,
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
        train_x = model.ops.asarray(train_x)
        train_y = model.ops.asarray(train_y)

        batches = model.ops.multibatch(batch_size, train_x, train_y, shuffle=True)

        for X, Y in batches:
            predictions, backprop = self.model.begin_update(X)
            gradient = predictions - Y
            loss = (gradient ** 2).sum(axis=1).mean()

            backprop(gradient)
            if sgd is not None:
                self.model.finish_update(sgd)
            losses[self.name] += loss

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
            for example in get_examples(nlp):
                pairs = example.reference._.rel
                for pair in pairs:
                    for label in pairs[pair]["relation"]:
                        self.add_label(label)

        subbatch = list(islice(get_examples(nlp), 10))
        doc_sample = [eg.reference for eg in subbatch]

        train_x = []
        train_y = []

        for doc in doc_sample:
            for pair in doc._.rel:
                train_x.append(doc._.rel[pair]["tensor"])
                train_y.append(np.array(list(doc._.rel[pair]["relation"].values())))

        train_x = np.array(train_x).astype(np.float32)
        train_y = np.array(train_y).astype(np.float32)
        train_x = model.ops.asarray(train_x)
        train_y = model.ops.asarray(train_y)

        self.model.initialize(X=train_x, Y=train_y)

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
        "rel_micro_p": precision,
        "rel_micro_r": recall,
        "rel_micro_f": f_score,
    }


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

    # for example in docbin_reader_dev(lang):
    #     doc = example.reference
    #     print(doc.has_extension("rel"))

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

    # rel_doc = rel_pipe(test_doc)
    # for pair in rel_doc._.rel:
    #     text = rel_doc._.rel[pair]["tuple"]
    #     predictions = rel_doc._.rel[pair]["relation"]
    #     print(f"{text} | {predictions}")
