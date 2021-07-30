<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Healthsea 🦑✨

This project uses NER & TEXTCAT to detect and classify health conditions in supplement reviews.

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
| `requirements` | Install dependencies and requirements |
| `generate_ner` | Generate data for NER annotation |
| `annotate_manual_ner` | Manually annotate data for NER training |
| `annotate_correct_ner` | Correct annotation data for NER training |
| `export_ner` | Export NER datasets from prodigy and convert to .spacy format |
| `train_ner` | Train a named entity recognition model |
| `evaluate_ner` | Evaluate a named entity recognition model |
| `generate_textcat` | Generate data for textcat annotation |
| `annotate_manual_textcat` | Manually annotate data for TEXTCAT training |
| `annotate_correct_textcat` | Correct annotation data for TEXTCAT training |
| `export_textcat` | Export TEXTCAT datasets from prodigy |
| `convert_textcat` | Convert exported TEXTCAT dataset to .spacy format |
| `train_textcat` | Train a text classification model |
| `evaluate_textcat` | Evaluate a text classification model |
| `assemble` | Assemble healthsea pipeline |
| `evaluate` | Evaluate healthsea pipeline |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `requirements` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/textcat/annotation.jsonl` | Local | Annotation data exported from Prodigy |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

---


