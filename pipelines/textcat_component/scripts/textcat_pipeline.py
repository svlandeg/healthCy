from itertools import islice
from typing import Iterable, Tuple, Optional, Dict, List, Callable, Any
from thinc.api import get_array_module, Model, Optimizer, set_dropout_rate, Config
from thinc.types import Floats2d
import numpy

from spacy.pipeline import TrainablePipe
from spacy.language import Language
from spacy.training import Example, validate_examples, validate_get_examples
from spacy.errors import Errors
from spacy.scorer import Scorer
from spacy.tokens import Span, Doc, Token
from spacy.vocab import Vocab

single_label_default_config = """
[model]
@architectures = "spacy.TextCatEnsemble.v2"
[model.tok2vec]
@architectures = "spacy.Tok2Vec.v2"
[model.tok2vec.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 64
rows = [2000, 2000, 1000, 1000, 1000, 1000]
attrs = ["ORTH", "LOWER", "PREFIX", "SUFFIX", "SHAPE", "ID"]
include_static_vectors = false
[model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = ${model.tok2vec.embed.width}
window_size = 1
maxout_pieces = 3
depth = 2
[model.linear_model]
@architectures = "spacy.TextCatBOW.v2"
exclusive_classes = true
ngram_size = 1
no_output_layer = false
"""
DEFAULT_SINGLE_TEXTCAT_MODEL = Config().from_str(single_label_default_config)["model"]

single_label_bow_config = """
[model]
@architectures = "spacy.TextCatBOW.v2"
exclusive_classes = true
ngram_size = 1
no_output_layer = false
"""

single_label_cnn_config = """
[model]
@architectures = "spacy.TextCatCNN.v2"
exclusive_classes = true
[model.tok2vec]
@architectures = "spacy.HashEmbedCNN.v2"
pretrained_vectors = null
width = 96
depth = 4
embed_size = 2000
window_size = 1
maxout_pieces = 3
subword_features = true
"""


@Language.factory(
    "textcat_healthsea",
    assigns=["doc._.statements"],
    default_config={"threshold": 0.5, "model": DEFAULT_SINGLE_TEXTCAT_MODEL},
    default_score_weights={
        "cats_score": 1.0,
        "cats_score_desc": None,
        "cats_micro_p": None,
        "cats_micro_r": None,
        "cats_micro_f": None,
        "cats_macro_p": None,
        "cats_macro_r": None,
        "cats_macro_f": None,
        "cats_macro_auc": None,
        "cats_f_per_type": None,
        "cats_macro_auc_per_type": None,
    },
)
def make_textcat(
    nlp: Language, name: str, model: Model[List[Doc], List[Floats2d]], threshold: float
) -> "TextCategorizer":
    """Create a TextCategorizer component. The text categorizer predicts categories
    over a whole document. It can learn one or more labels, and the labels are considered
    to be mutually exclusive (i.e. one true label per doc).
    model (Model[List[Doc], List[Floats2d]]): A model instance that predicts
        scores for each category.
    threshold (float): Cutoff to consider a prediction "positive".
    """
    return TextCategorizer(nlp.vocab, model, name, threshold=threshold)


class TextCategorizer(TrainablePipe):
    """Pipeline component for single-label text classification.
    DOCS: https://spacy.io/api/textcategorizer
    """

    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "textcat_healthsea",
        *,
        threshold: float,
    ) -> None:
        """Initialize a text categorizer for single-label classification.
        vocab (Vocab): The shared vocabulary.
        model (thinc.api.Model): The Thinc Model powering the pipeline component.
        name (str): The component instance name, used to add entries to the
            losses during training.
        threshold (float): Cutoff to consider a prediction "positive".
        DOCS: https://spacy.io/api/textcategorizer#init
        """
        self.vocab = vocab
        self.model = model
        self.name = name
        self._rehearsal_model = None
        cfg = {"labels": [], "threshold": threshold, "positive_label": None}
        self.cfg = dict(cfg)

    @property
    def labels(self) -> Tuple[str]:
        """RETURNS (Tuple[str]): The labels currently added to the component.
        DOCS: https://spacy.io/api/textcategorizer#labels
        """
        return tuple(self.cfg["labels"])

    @property
    def label_data(self) -> List[str]:
        """RETURNS (List[str]): Information about the component's labels.
        DOCS: https://spacy.io/api/textcategorizer#label_data
        """
        return self.labels

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipe to a Doc."""
        predictions = self.predict_statements([doc])
        self.set_annotations_statements([doc], predictions)
        return doc

    def predict_statements(self, docs: Iterable[Doc]):
        statement_list = []
        for doc in docs:
            statements = extract_clauses(doc)
            statement_list += statements

        statement_docs = [statement[0] for statement in statement_list]

        scores = self.model.predict(statement_docs)
        scores = self.model.ops.asarray(scores)
        return scores

    def set_annotations_statements(self, docs: Iterable[Doc], scores) -> None:
        index = 0
        for doc in docs:
            statements = extract_clauses(doc)
            classified_statements = []
            for statement in statements:
                prediction = scores[index]
                cats = {}

                for value, label in zip(prediction, self.labels):
                    cats[label] = value

                classified_statements.append((statement[0], statement[1], cats))
                index += 1
            doc.set_extension("statements", default=[], force=True)
            doc._.statements = classified_statements

    def predict(self, docs: Iterable[Doc]):
        """Apply the pipeline's model to a batch of docs, without modifying them.
        docs (Iterable[Doc]): The documents to predict.
        RETURNS: The models prediction for each document.
        DOCS: https://spacy.io/api/textcategorizer#predict
        """
        if not any(len(doc) for doc in docs):
            # Handle cases where there are no tokens in any docs.
            tensors = [doc.tensor for doc in docs]
            xp = get_array_module(tensors)
            scores = xp.zeros((len(docs), len(self.labels)))
            return scores
        scores = self.model.predict(docs)
        scores = self.model.ops.asarray(scores)
        return scores

    def set_annotations(self, docs: Iterable[Doc], scores) -> None:
        """Modify a batch of Doc objects, using pre-computed scores.
        docs (Iterable[Doc]): The documents to modify.
        scores: The scores to set, produced by TextCategorizer.predict.
        DOCS: https://spacy.io/api/textcategorizer#set_annotations
        """
        for i, doc in enumerate(docs):
            for j, label in enumerate(self.labels):
                doc.cats[label] = float(scores[i, j])

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to predict and get_loss.
        examples (Iterable[Example]): A batch of Example objects.
        drop (float): The dropout rate.
        sgd (thinc.api.Optimizer): The optimizer.
        losses (Dict[str, float]): Optional record of the loss during training.
            Updated using the component name as the key.
        RETURNS (Dict[str, float]): The updated losses dictionary.
        DOCS: https://spacy.io/api/textcategorizer#update
        """
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        validate_examples(examples, "TextCategorizer.update")
        self._validate_categories(examples)
        if not any(len(eg.predicted) if eg.predicted else 0 for eg in examples):
            # Handle cases where there are no tokens in any docs.
            return losses
        set_dropout_rate(self.model, drop)
        scores, bp_scores = self.model.begin_update([eg.predicted for eg in examples])
        loss, d_scores = self.get_loss(examples, scores)
        bp_scores(d_scores)
        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += loss
        return losses

    def rehearse(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Perform a "rehearsal" update from a batch of data. Rehearsal updates
        teach the current model to make predictions similar to an initial model,
        to try to address the "catastrophic forgetting" problem. This feature is
        experimental.
        examples (Iterable[Example]): A batch of Example objects.
        drop (float): The dropout rate.
        sgd (thinc.api.Optimizer): The optimizer.
        losses (Dict[str, float]): Optional record of the loss during training.
            Updated using the component name as the key.
        RETURNS (Dict[str, float]): The updated losses dictionary.
        DOCS: https://spacy.io/api/textcategorizer#rehearse
        """
        if losses is not None:
            losses.setdefault(self.name, 0.0)
        if self._rehearsal_model is None:
            return losses
        validate_examples(examples, "TextCategorizer.rehearse")
        self._validate_categories(examples)
        docs = [eg.predicted for eg in examples]
        if not any(len(doc) for doc in docs):
            # Handle cases where there are no tokens in any docs.
            return losses
        set_dropout_rate(self.model, drop)
        scores, bp_scores = self.model.begin_update(docs)
        target = self._rehearsal_model(examples)
        gradient = scores - target
        bp_scores(gradient)
        if sgd is not None:
            self.finish_update(sgd)
        if losses is not None:
            losses[self.name] += (gradient ** 2).sum()
        return losses

    def _examples_to_truth(
        self, examples: List[Example]
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        truths = numpy.zeros((len(examples), len(self.labels)), dtype="f")
        not_missing = numpy.ones((len(examples), len(self.labels)), dtype="f")
        for i, eg in enumerate(examples):
            for j, label in enumerate(self.labels):
                if label in eg.reference.cats:
                    truths[i, j] = eg.reference.cats[label]
                else:
                    not_missing[i, j] = 0.0
        truths = self.model.ops.asarray(truths)
        return truths, not_missing

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores.
        examples (Iterable[Examples]): The batch of examples.
        scores: Scores representing the model's predictions.
        RETURNS (Tuple[float, float]): The loss and the gradient.
        DOCS: https://spacy.io/api/textcategorizer#get_loss
        """
        validate_examples(examples, "TextCategorizer.get_loss")
        self._validate_categories(examples)
        truths, not_missing = self._examples_to_truth(examples)
        not_missing = self.model.ops.asarray(not_missing)
        d_scores = (scores - truths) / scores.shape[0]
        d_scores *= not_missing
        mean_square_error = (d_scores ** 2).sum(axis=1).mean()
        return float(mean_square_error), d_scores

    def add_label(self, label: str) -> int:
        """Add a new label to the pipe.
        label (str): The label to add.
        RETURNS (int): 0 if label is already present, otherwise 1.
        DOCS: https://spacy.io/api/textcategorizer#add_label
        """
        if not isinstance(label, str):
            raise ValueError(Errors.E187)
        if label in self.labels:
            return 0
        self._allow_extra_label()
        self.cfg["labels"].append(label)
        if self.model and "resize_output" in self.model.attrs:
            self.model = self.model.attrs["resize_output"](
                self.model, len(self.cfg["labels"])
            )
        self.vocab.strings.add(label)
        return 1

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Optional[Language] = None,
        labels: Optional[Iterable[str]] = None,
        positive_label: Optional[str] = None,
    ) -> None:
        """Initialize the pipe for training, using a representative set
        of data examples.
        get_examples (Callable[[], Iterable[Example]]): Function that
            returns a representative sample of gold-standard Example objects.
        nlp (Language): The current nlp object the component is part of.
        labels (Optional[Iterable[str]]): The labels to add to the component, typically generated by the
            `init labels` command. If no labels are provided, the get_examples
            callback is used to extract the labels from the data.
        positive_label (Optional[str]): The positive label for a binary task with exclusive classes,
            `None` otherwise and by default.
        DOCS: https://spacy.io/api/textcategorizer#initialize
        """
        validate_get_examples(get_examples, "TextCategorizer.initialize")
        self._validate_categories(get_examples())
        if labels is None:
            for example in get_examples():
                for cat in example.y.cats:
                    self.add_label(cat)
        else:
            for label in labels:
                self.add_label(label)
        if len(self.labels) < 2:
            raise ValueError(Errors.E867)
        if positive_label is not None:
            if positive_label not in self.labels:
                err = Errors.E920.format(pos_label=positive_label, labels=self.labels)
                raise ValueError(err)
            if len(self.labels) != 2:
                err = Errors.E919.format(pos_label=positive_label, labels=self.labels)
                raise ValueError(err)
        self.cfg["positive_label"] = positive_label
        subbatch = list(islice(get_examples(), 10))
        doc_sample = [eg.reference for eg in subbatch]
        label_sample, _ = self._examples_to_truth(subbatch)
        self._require_labels()
        assert len(doc_sample) > 0, Errors.E923.format(name=self.name)
        assert len(label_sample) > 0, Errors.E923.format(name=self.name)
        self.model.initialize(X=doc_sample, Y=label_sample)

    def score(self, examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
        """Score a batch of examples.
        examples (Iterable[Example]): The examples to score.
        RETURNS (Dict[str, Any]): The scores, produced by Scorer.score_cats.
        DOCS: https://spacy.io/api/textcategorizer#score
        """
        validate_examples(examples, "TextCategorizer.score")
        self._validate_categories(examples)
        kwargs.setdefault("threshold", self.cfg["threshold"])
        kwargs.setdefault("positive_label", self.cfg["positive_label"])
        return Scorer.score_cats(
            examples,
            "cats",
            labels=self.labels,
            multi_label=False,
            **kwargs,
        )

    def _validate_categories(self, examples: List[Example]):
        """Check whether the provided examples all have single-label cats annotations."""
        for ex in examples:
            if list(ex.reference.cats.values()).count(1.0) > 1:
                raise ValueError(Errors.E895.format(value=ex.reference.cats))


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
