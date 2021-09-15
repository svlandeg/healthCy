import pytest
from typing import Callable, List, Tuple
import spacy
from textcat_clauses import textcat_clauses
from spacy.vocab import Vocab
from spacy.tokens import Doc, Span
from spacy.language import Language
from thinc.api import Model
from thinc.types import Ragged
import typer
from pathlib import Path
import benepar

# Initialize model
def generate_textcat_model(vocab):
    nlp = Language(vocab)
    textcat = nlp.add_pipe("textcat")
    textcat.add_label("NEUTRAL")
    textcat.add_label("ANAMNESIS")
    textcat.add_label("POSITIVE")
    textcat.add_label("NEGATIVE")
    nlp.initialize()
    return textcat.model


def generate_textcat_clauses_model(textcat_model, get_clauses, docs):
    model = textcat_clauses(textcat_model, get_clauses=get_clauses)
    model.initialize(X=docs)
    return model


# Clause Segmentation
def split_sentence_berkely(sentence: Span) -> List[Span]:
    split_sentences = []
    for constituent in sentence._.constituents:
        if "S" in constituent._.labels and constituent._.parent == sentence:
            split_sentences.append(constituent)
    if len(split_sentences) == 0:
        split_sentences.append(sentence)
    return split_sentences


def construct_statement(clauses: List[Span]) -> List[Tuple[Doc, List[Span]]]:
    statement_list = []
    for clause in clauses:
        if len(clause.ents) > 0:
            for index in range(len(clause.ents)):
                start = clause.ents[index].start
                end = clause.ents[index].end

                words = []
                replaced = False

                for word in clause:
                    if word.i >= start and word.i < end and not replaced:
                        words.append(f"<{clause.ents[index].label_}>")
                        replaced = True
                    elif not (word.i >= start and word.i < end):
                        words.append(word.text)

                doc = Doc(clause.doc.vocab, words=words)
                entity = str(clause.ents[index]).lower().strip()
                entity = entity.replace(" ", "_")

                statement_list.append((doc, entity, clause.ents[index].label_))
        else:
            words = [word.text for word in clause]
            doc = Doc(clause.doc.vocab, words=words)
            statement_list.append((doc, None, None))

    return statement_list


def extract_clauses(doc: Doc) -> List[Tuple[Doc, Span]]:
    return_list = []
    for sentence in doc.sents:
        split_clauses = split_sentence_berkely(sentence)
        statements = construct_statement(split_clauses)
        return_list += statements
    return return_list


# Testing
def textcat_clauses_test(ner_model: Path):
    nlp = spacy.load(ner_model)
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    texts = [
        "This helps my <CONDITION>",
        "I bought this yesterday. It really helps my headache!",
        "Great product",
    ]
    docs = [nlp(doc) for doc in texts]
    textcat_model = generate_textcat_model(nlp.vocab)
    textcat_clause_model = generate_textcat_clauses_model(
        textcat_model, extract_clauses, docs
    )

    scores, backprop = textcat_clause_model(docs, is_train=True)

    for doc_i in range(len(docs)):
        doc_clauses = extract_clauses(docs[doc_i])
        print(doc_clauses, scores[doc_i])


if __name__ == "__main__":
    typer.run(textcat_clauses_test)
