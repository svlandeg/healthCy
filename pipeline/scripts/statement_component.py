import spacy
from spacy.tokens import Span, Doc, Token
from spacy.language import Language
from typing import Tuple, List


@Language.factory("statement_classification", default_config={})
def create_statement_classification(nlp: Language, name: str):
    return Statementclassification(nlp)


class Statementclassification:
    def __init__(self, nlp: Language):
        self.nlp = nlp

    def __call__(self, doc: Doc):
        transformers = self.nlp.get_pipe("transformer_textcat")
        textcat = self.nlp.get_pipe("textcat")

        statements = extract_clauses(doc)
        statements_list = []
        for statement in statements:
            classification_doc = textcat(transformers(statement[0]))
            classification = classification_doc.cats
            statements_list.append((statement[0], statement[1], classification))

        doc.set_extension("statements", default=[], force=True)
        doc._.statements = statements_list

        return doc


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

    if len(verb_chunks) > 1:
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


def construct_statement(clauses: Span) -> List[Tuple[Doc, List[Span]]]:

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
                statement_list.append((doc, clause.ents[index]))
        else:
            words = [word.text for word in clause]
            doc = Doc(clause.doc.vocab, words=words)
            statement_list.append((doc, []))

    return statement_list


def extract_clauses(doc: Doc) -> List[Tuple[Doc, Span]]:
    return_list = []
    for sentence in doc.sents:
        verb_chunks = get_verb_chunk(sentence)
        split_clauses = split_sentence(sentence, verb_chunks)
        statements = construct_statement(split_clauses)
        return_list += statements
    return return_list
