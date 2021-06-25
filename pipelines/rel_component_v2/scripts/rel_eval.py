import spacy
import typer
from pathlib import Path

# make the factory work
from rel_pipeline import make_relation_extractor

# make the config work
from rel_model import create_relation_model, create_classification_layer

if __name__ == "__main__":

    spacy.prefer_gpu()

    nlp = spacy.load("training/model-best")
    text = "This helped my joint pain barely"

    doc = nlp(text)
    relations = doc._.rel

    for ents in doc.ents:
        print(ents)

    for relation in relations:
        if relations[relation]["relation"]["RELATED"] == 1:
            print(relations[relation]["tuple"])
