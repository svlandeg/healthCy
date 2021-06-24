from functools import partial
from pathlib import Path
from typing import Iterable, Callable
import spacy
from spacy.training import Example
from spacy.tokens import DocBin, Doc
import copy


@spacy.registry.readers("Gold_ents_Corpus.v1")
def create_docbin_reader(file: Path) -> Callable[["Language"], Iterable[Example]]:
    return partial(read_files, file)


def read_files(file: Path, nlp: "Language") -> Iterable[Example]:
    doc_bin = DocBin().from_disk(file)
    docs = doc_bin.get_docs(nlp.vocab)

    for gold in docs:
        pred = Doc(
            nlp.vocab,
            words=[t.text for t in gold],
            spaces=[t.whitespace_ for t in gold],
            tags=[t.tag_ for t in gold],
            pos=[t.pos_ for t in gold],
            deps=[t.dep_ for t in gold],
        )

        pred.ents = gold.ents
        pred._.rel = copy.deepcopy(gold._.rel)

        yield Example(pred, gold)


if __name__ == "__main__":
    from spacy.lang.en import English
    from spacy.tokens import Span, DocBin, Doc

    file = "../data/train.spacy"
    lang = English()
    Doc.set_extension("rel", default={})

    docbin_reader = create_docbin_reader(file)
    data = docbin_reader(lang)

    print(data[0].text)
