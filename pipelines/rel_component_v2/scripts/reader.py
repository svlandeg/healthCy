from functools import partial
from pathlib import Path
from typing import Iterable, Callable
import spacy
from spacy.training import Example
from spacy.tokens import DocBin, Doc


@spacy.registry.readers("Gold_ents_Corpus.v1")
def create_docbin_reader(file: Path) -> Callable[["Language"], Iterable[Example]]:
    return partial(read_files, file)


def read_files(file: Path, nlp: "Language") -> Iterable[Example]:
    doc_bin = DocBin().from_disk(file)
    docs = doc_bin.get_docs(nlp.vocab)

    for gold in docs:
        pred = gold
        for pair in pred._.rel:
            for key in pred._.rel[pair]["relation"]:
                pred._.rel[pair]["relation"][key] = 0

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