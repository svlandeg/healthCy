import spacy
from spacy import displacy


nlp = spacy.load("../training/model-best")

example = "This helped my joint pain"
doc = nlp(example)

for token in doc:
    print(f"{token.text} | {token.ent_type_} | {token.dep_} | {token.pos_}")
