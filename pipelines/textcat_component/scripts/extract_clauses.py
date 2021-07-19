import spacy
from spacy.tokens import Span, Doc, Token

from typing import Tuple, List, Iterable, Optional, Dict, Callable, Any

import typer
from pathlib import Path

import json
import secrets


def main(ner_model: Path, examples: Path, gpu: bool):

    if gpu:
        spacy.prefer_gpu()

    nlp = spacy.load(ner_model)

    example_reviews = []
    with open(examples) as f:
        example_reviews = json.load(f)
        f.close()

    example_batch = []
    amount = 25
    for i in range(amount):
        example_batch.append(secrets.choice(example_reviews))

    example_batch = [
        "This helped joint pain barely",
        "This helped asthma but made inflammation worse",
        "This is great for energy, muscle growth and bone health",
        "This increases energy and decreases fatigue",
        "This helps hair, nails and skin to grow stronger",
    ]

    for example in example_batch:
        # review = example["text"]
        review = example
        doc = nlp(review)

        statements = extract_clauses(doc)
        print(doc)
        for statement in statements:
            print(f"  >> {statement}")
        print()


def get_verb_chunk(sentence: Span) -> List[List[Token]]:
    verb_chunks = []
    last_chunk = 0
    for word in sentence:
        if (
            (word.pos_ == "VERB" or word.pos_ == "AUX")
            and word.i + 1 < len(word.doc)
            and word.i > last_chunk
        ):
            verb_chunk = [word]
            for wordx2 in sentence[word.i + 1 :]:
                if (
                    wordx2.pos_ == "VERB"
                    or wordx2.pos_ == "AUX"
                    or wordx2.pos_ == "PART"
                ):
                    verb_chunk.append(wordx2)
                else:
                    break

            verb_chunks.append(verb_chunk)
            last_chunk = verb_chunk[-1].i

    return verb_chunks


def split_sentence(sentence: Span, verb_chunks: List[List[Token]]) -> List[Span]:

    split_triggers = ["CCONJ"]
    split_indices = []
    split_sentences = []
    sentence_boundaries = [sentence[0].i, sentence[-1].i]

    if len(verb_chunks) <= 1:
        return [sentence]

    for i in range(0, len(verb_chunks) - 1):
        start = verb_chunks[i][-1].i
        end = verb_chunks[i + 1][0].i

        if start + 1 == end:
            continue

        for index in range(end, start, -1):
            if sentence.doc[index].pos_ in split_triggers:
                split_indices.append(sentence.doc[index].i)
                break

    if len(split_indices) > 0:
        lastIndex = sentence_boundaries[0]
        for i in range(0, len(split_indices)):
            split_sentences.append(sentence.doc[lastIndex : split_indices[i]])
            lastIndex = split_indices[i]
        split_sentences.append(sentence.doc[lastIndex + 1 : sentence_boundaries[1] + 1])

    else:
        return [sentence]

    return split_sentences


def construct_statement(clauses: Span) -> List[Tuple[str, Span]]:

    statement_list = []
    for clause in clauses:
        for index in range(len(clause.ents)):
            clause_text = str(clause)
            clause_text = clause_text.replace(
                str(clause.ents[index]), f"<{clause.ents[index].label_}>"
            )
            clause_text = clause_text.strip()
            statement_list.append((clause_text, clause.ents[index]))
    return statement_list


def extract_clauses(doc: Doc) -> List[Tuple[str, Span]]:
    return_list = []
    for sentence in doc.sents:
        verb_chunks = get_verb_chunk(sentence)
        split_clauses = split_sentence(sentence, verb_chunks)
        return_list += construct_statement(split_clauses)
    return return_list


if __name__ == "__main__":
    typer.run(main)
