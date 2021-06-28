<a href="https://www.youtube.com/watch?v=8HL-Ap5_Axo" target="_blank"><img src="https://user-images.githubusercontent.com/8796347/117116338-8566cc00-ad8e-11eb-9cd3-e88e94fadb6a.jpg" width="300" height="auto" align="right" /></a>


<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Example project of relation extraction

This example project shows how to implement a spaCy component with a custom Machine Learning model.

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
| `data_cpu` | Parse the gold-standard annotations from the Prodigy annotations. |
| `data_gpu` | Parse the gold-standard annotations from the Prodigy annotations. |
| `train_cpu` | Train the REL model on the CPU and evaluate on the dev corpus. |
| `train_gpu` | Train the REL model on the GPU and evaluate on the dev corpus. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all_cpu` | `data_cpu` &rarr; `train_cpu` |
| `all_gpu` | `data_gpu` &rarr; `train_gpu` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/annotations.jsonl`](assets/annotations.jsonl) | Local | Gold-standard REL annotations created with Prodigy |
| [`assets/dependencies.json`](assets/dependencies.json) | Local | List of dependency tags |
| [`assets/partofspeech.json`](assets/partofspeech.json) | Local | List of part-of-speech tags |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
