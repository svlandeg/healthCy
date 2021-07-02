from typing import List

import spacy
from spacy.tokens import Doc, DocBin
from thinc.types import Floats2d
from thinc.api import Model, Linear, Relu, chain, Logistic, Softmax, Sigmoid


@spacy.registry.architectures("rel_model.v2")
def create_relation_model(
    classification_layer: Model[Floats2d, Floats2d],
) -> Model[List[Doc], Floats2d]:
    with Model.define_operators({">>": chain}):
        model = classification_layer
    return model


@spacy.registry.architectures("rel_classification_layer.v2")
def create_classification_layer(
    nO: int = None, nI: int = None
) -> Model[Floats2d, Floats2d]:
    with Model.define_operators({">>": chain}):
        return (
            Linear(nO=516, nI=nI)
            >> Linear(nO=128, nI=516)
            >> Linear(nO=nO, nI=128)
            >> Logistic()
        )


if __name__ == "__main__":

    from spacy.vocab import Vocab
    import numpy as np
    from thinc.api import prefer_gpu
    import thinc.util
    from thinc.api import Adam, fix_random_seed
    from tqdm.notebook import tqdm
    from spacy.scorer import PRFScore
    from wasabi import Printer

    msg = Printer()
    vocab = Vocab()
    Doc.set_extension("rel", default={})

    training_path = "../data/train.spacy"
    training_data = list(DocBin().from_disk(training_path).get_docs(vocab))

    dev_path = "../data/dev.spacy"
    dev_data = list(DocBin().from_disk(dev_path).get_docs(vocab))

    train_x = []
    train_y = []

    dev_x = []
    dev_y = []

    relations = list(
        training_data[0]
        ._.rel[list(training_data[0]._.rel.keys())[0]]["relation"]
        .keys()
    )

    for doc in training_data:
        for pairs in doc._.rel:
            train_x.append(doc._.rel[pairs]["tensor"])
            train_y.append(np.array(list(doc._.rel[pairs]["relation"].values())))

    for doc in dev_data:
        for pairs in doc._.rel:
            dev_x.append(doc._.rel[pairs]["tensor"])
            dev_y.append(np.array(list(doc._.rel[pairs]["relation"].values())))

    msg.info(f"Train docs: {len(training_data)} | Dev docs: {len(dev_data)}")
    msg.info(f"Train size: {len(train_x)} | Dev size : {len(dev_x)}")
    msg.info(f"Output classes: {relations}")

    train_x = np.array(train_x).astype(np.float32)
    train_y = np.array(train_y).astype(np.float32)
    dev_x = np.array(dev_x).astype(np.float32)
    dev_y = np.array(dev_y).astype(np.float32)

    model = create_relation_model(create_classification_layer(None, None))

    train_x = model.ops.asarray(train_x)
    train_y = model.ops.asarray(train_y)
    dev_x = model.ops.asarray(dev_x)
    dev_y = model.ops.asarray(dev_y)

    model.initialize(X=train_x[:5], Y=train_y[:5])
    nI = model.get_dim("nI")
    nO = model.get_dim("nO")
    msg.good(
        f"Initialized model with input dimension nI={nI} and output dimension nO={nO}"
    )

    fix_random_seed(0)
    optimizer = Adam(0.001)
    batch_size = 128
    epochs = 100
    threshold = 0.4
    msg.text("Start training")

    for i in range(epochs):
        batches = model.ops.multibatch(batch_size, train_x, train_y, shuffle=True)
        i_loss = 0
        for X, Y in batches:
            Yh, backprop = model.begin_update(X)

            gradient = Yh - Y
            loss = (gradient ** 2).sum(axis=1).mean()
            backprop(gradient)
            model.finish_update(optimizer)
            i_loss += float(loss)

        score_dict = {}

        for X, Y in model.ops.multibatch(batch_size, dev_x, dev_y):
            Yh = model.predict(X)

            for output_yh, output_y in zip(Yh, Y):
                max_val = np.max(output_yh)

                for pred, expected, relation in zip(output_yh, output_y, relations):
                    if relation not in score_dict:
                        score_dict[relation] = PRFScore()

                    prediction = 0
                    if pred >= threshold and pred == max_val:
                        prediction = 1

                    if prediction == 1 and expected == 1:
                        score_dict[relation].tp += 1
                    elif prediction == 1 and expected == 0:
                        score_dict[relation].fp += 1
                    elif prediction == 0 and expected == 1:
                        score_dict[relation].fn += 1

        global_fscore = 0
        global_precision = 0
        global_recall = 0

        for relation in score_dict:
            msg.info(
                f" {i} | Class: {relation} F-Score:{float(score_dict[relation].fscore):.3f}, Precision:{float(score_dict[relation].precision):.3f}, Recall:{float(score_dict[relation].recall):.3f}"
            )

            global_fscore += float(score_dict[relation].fscore)
            global_precision += float(score_dict[relation].precision)
            global_recall += float(score_dict[relation].recall)

        global_fscore /= len(score_dict)
        global_precision /= len(score_dict)
        global_recall /= len(score_dict)

        msg.info(
            f" {i} | Loss: {i_loss:.3f}, F-Score:{float(global_fscore):.3f}, Precision:{float(global_precision):.3f}, Recall:{float(global_recall):.3f}"
        )

        print("")
