import spacy
from spacy import displacy

nlp = spacy.load("../training/model-best")

example = "This helped my joint pain"
doc = nlp(example)

displacy.serve(doc, style="dep")
displacy.serve(doc, style="ent")
