import spacy
from spacy.tokens import Span, Doc, Token
from spacy.language import Language
from typing import Tuple, List
import operator
import benepar


@Language.factory("statement_classification", default_config={})
def create_statement_classification(nlp: Language, name: str):
    return Statementclassification(nlp)


class Statementclassification:
    def __init__(self, nlp: Language):
        self.nlp = nlp

    def __call__(self, doc: Doc):

        if self.nlp.has_pipe("transformer_textcat"):
            tok2vec = self.nlp.get_pipe("transformer_textcat")

        elif self.nlp.has_pipe("tok2vec_textcat"):
            tok2vec = self.nlp.get_pipe("tok2vec_textcat")

        textcat = self.nlp.get_pipe("textcat")

        # Clause Segmentation
        statements = extract_clauses(doc)
        statements_list = []
        for statement in statements:
            classification_doc = textcat(tok2vec(statement[0]))
            classification = classification_doc.cats

            statements_list.append(
                (statement[0], statement[1], classification, statement[2])
            )

        doc.set_extension("statements", default=[], force=True)
        doc._.statements = statements_list

        # Patient Information
        patient_information = []
        for statement in statements_list:
            classification = max(statement[2].items(), key=operator.itemgetter(1))[0]
            entity = statement[1]

            if classification == "ANAMNESIS" and entity != None:
                patient_information.append((entity, []))
            else:
                for patient_health in patient_information:
                    patient_health[1].append(classification)

        doc.set_extension("patient_information", default=[], force=True)
        doc._.patient_information = patient_information

        # Condition/Benefit summary
        effect_summary = {}
        for statement in statements_list:
            classification = max(statement[2].items(), key=operator.itemgetter(1))[0]
            entity = statement[1]

            if entity != None:
                if entity not in effect_summary:
                    effect_summary[entity] = {
                        "classification": [],
                        "label": statement[3],
                    }
                effect_summary[entity]["classification"].append(classification)

        for patient_health in patient_information:
            entity = patient_health[0]
            score = 0
            end_classification = "NEUTRAL"
            for classification in patient_health[1]:
                if classification == "POSITIVE":
                    score += 1
                elif classification == "NEGATIVE":
                    score -= 1

            if score > 0:
                end_classification = "POSITIVE"
            elif score < 0:
                end_classification = "NEGATIVE"

            effect_summary[entity]["classification"].append(classification)

        effect_summary_unique = {}
        for effect_index in effect_summary:
            score = 0
            end_classification = "NEUTRAL"

            for classification in effect_summary[effect_index]["classification"]:
                if classification == "POSITIVE":
                    score += 1
                elif classification == "NEGATIVE":
                    score -= 1

            if score > 0:
                end_classification = "POSITIVE"
            elif score < 0:
                end_classification = "NEGATIVE"

            effect_summary_unique[effect_index] = {}
            effect_summary_unique[effect_index]["classification"] = end_classification
            effect_summary_unique[effect_index]["label"] = effect_summary[effect_index][
                "label"
            ]

        doc.set_extension("effect_summary", default=[], force=True)
        doc._.effect_summary = effect_summary_unique

        return doc


# Clause Segmentation


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
        verb_chunks = get_verb_chunk(sentence)
        # split_clauses = split_sentence(sentence, verb_chunks)
        split_clauses = split_sentence_berkely(sentence)
        statements = construct_statement(split_clauses)
        return_list += statements
    return return_list


if __name__ == "__main__":

    import benepar, spacy

    spacy.prefer_gpu()

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    doc = nlp(
        "It a very good rub , the smell is just perfect , i like to mix it with some essential oils and rub the chest or the feet of my kids when they take a cold."
    )

    split_sentences = []
    for sentence in doc.sents:
        split_sentences = []
        for constituent in sentence._.constituents:
            if "S" in constituent._.labels and constituent._.parent == sentence:
                split_sentences.append(constituent)
        if len(split_sentences) == 0:
            split_sentences.append(sentence)
    print(len(split_sentences), split_sentences)
    print()

    # spacy.prefer_gpu()
    # nlp = spacy.load("../training/healthsea/config_trf")
    # doc = nlp(
    #    "They helped my constipation barely but at least the improved my eye sight!"
    # )
    # print(doc._.statements)
