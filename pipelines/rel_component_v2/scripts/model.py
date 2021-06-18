from typing import List

import spacy
from spacy.tokens import Doc
from thinc.types import Floats2d,
from thinc.api import Model, Linear, chain, Logistic


@spacy.registry.architectures("rel_model.v2")
def create_relation_model(
    classification_layer: Model[Floats2d, Floats2d],
) -> Model[List[Doc], Floats2d]:
    with Model.define_operators({">>": chain}):
        model = classification_layer
    return model


@spacy.registry.architectures("rel_classification_layer.v1")
def create_classification_layer(
    nO: int = None, nI: int = None
) -> Model[Floats2d, Floats2d]:
    with Model.define_operators({">>": chain}):
        return Linear(nO=nO, nI=nI) >> Logistic()
