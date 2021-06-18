<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: HealthCy NER. Detect health conditions in supplement reviews.

This project trains a NER with the labels `CONDITION` and `BENEFIT`

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `install` | Install dependencies |
| `train` | Train a named entity recognition model |
| `train_gpu` | Train a named entity recognition model |
| `evaluate` | Evaluate the ner model and export metric |
| `evaluate_gpu` | Evaluate the ner model and export metric |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `install` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/train.spacy`](assets/train.spacy) | Local | Training data exported from Prodigy |
| [`assets/dev.spacy`](assets/dev.spacy) | Local | Development data exported from Prodigy |
| [`assets/test.spacy`](assets/test.spacy) | Local | Test data exported from Prodigy |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

---


