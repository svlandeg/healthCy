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


@spacy.registry.architectures("rel_classification_layer.v1")
def create_classification_layer(
    nO: int = None, nI: int = None, dropout: float = 0.2
) -> Model[Floats2d, Floats2d]:
    with Model.define_operators({">>": chain}):
        return (
            Linear(nO=nO, nI=nI)
            >> Linear(nO=nO, nI=nI)
            >> Linear(nO=nO, nI=nI)
            >> Logistic()
        )


if __name__ == "__main__":

    from spacy.vocab import Vocab
    import numpy as np
    from thinc.api import prefer_gpu
    import thinc.util
    from thinc.api import Adam, fix_random_seed
    from tqdm.notebook import tqdm

    print("Thinc GPU?", prefer_gpu())

    Doc.set_extension("rel", default={})
    vocab = Vocab()

    training_path = "../data/train.spacy"
    training_data = list(DocBin().from_disk(training_path).get_docs(vocab))

    dev_path = "../data/dev.spacy"
    dev_data = list(DocBin().from_disk(dev_path).get_docs(vocab))

    train_x = []
    train_y = []

    dev_x = []
    dev_y = []

    for doc in training_data:
        for pairs in doc._.rel:
            train_x.append(doc._.rel[pairs]["tensor"])
            train_y.append(np.array(list(doc._.rel[pairs]["relation"].values())))

    for doc in dev_data:
        for pairs in doc._.rel:
            dev_x.append(doc._.rel[pairs]["tensor"])
            dev_y.append(np.array(list(doc._.rel[pairs]["relation"].values())))

    print(f"Train docs: {len(training_data)} | Dev docs: {len(dev_data)}")
    print(f"Train size: {len(train_x)} | Dev size : {len(dev_x)}")

    train_x = np.array(train_x).astype(np.float32)
    train_y = np.array(train_y).astype(np.float32)
    dev_x = np.array(dev_x).astype(np.float32)
    dev_y = np.array(dev_y).astype(np.float32)

    print(train_x.dtype, train_y.dtype, dev_x.dtype, dev_y.dtype)

    model = create_relation_model(create_classification_layer(None, None))

    train_x = model.ops.asarray(train_x)
    train_y = model.ops.asarray(train_y)
    dev_x = model.ops.asarray(dev_x)
    dev_y = model.ops.asarray(dev_y)

    model.initialize(X=train_x[:5], Y=train_y[:5])
    nI = model.get_dim("nI")
    nO = model.get_dim("nO")
    print(
        f"Initialized model with input dimension nI={nI} and output dimension nO={nO}"
    )

    fix_random_seed(0)
    optimizer = Adam(0.001)
    batch_size = 128
    epochs = 100
    threshold = 0.7
    print("Measuring performance across iterations:")

    for i in range(epochs):
        batches = model.ops.multibatch(batch_size, train_x, train_y, shuffle=True)
        for X, Y in batches:
            Yh, backprop = model.begin_update(X)
            backprop(Yh - Y)
            model.finish_update(optimizer)
        # Evaluate and print progress
        correct = 0

        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        total = 0
        for X, Y in model.ops.multibatch(batch_size, dev_x, dev_y):
            Yh = model.predict(X)

            for k in range(0, len(Y)):
                prediction = 0
                if Yh[k] >= threshold:
                    prediction = 1

                if prediction == 1 and Y[k] == 1:
                    true_positive += 1
                elif prediction == 1 and Y[k] == 0:
                    false_positive += 1
                elif prediction == 0 and Y[k] == 0:
                    true_negative += 1
                elif prediction == 0 and Y[k] == 1:
                    false_negative += 1

            total += len(Y)

            # correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
            # total += Yh.shape[0]

        precision = 0
        if (true_positive + false_positive) != 0:
            precision = true_positive / (true_positive + false_positive)

        recall = 0
        if (true_positive + false_negative) != 0:
            recall = true_positive / (true_positive + false_negative)

        f_score = 0
        if (precision + recall) != 0:
            f_score = (2 * precision * recall) / (precision + recall)

        print(
            f" {i} F-Score:{float(f_score):.3f}, Precision:{float(precision):.3f}, Recall:{float(recall):.3f}"
        )
