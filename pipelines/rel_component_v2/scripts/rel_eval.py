import spacy
import typer
from pathlib import Path

# make the factory work
from rel_pipeline import make_relation_extractor

# make the config work
from rel_model import create_relation_model, create_classification_layer

if __name__ == "__main__":

    spacy.prefer_gpu()

    nlp = spacy.load("training/model-best")
    texts = [
        "I have heart ache",
        "This helped my joint pain barely",
        "This improves asthma but causes headaches",
        "It makes my skin glow",
        "Feeling nausea after taking",
        "Gives me more energy and reduces fatigue",
    ]

    for text in texts:
        doc = nlp(text)
        relations = doc._.rel

        for ents in doc.ents:
            ent_key = (ents.start, ents.end - 1)
            entity = ents.text
            label = ents.label_
            t_before = ""
            t_after = ""

            for r_key in relations:
                if relations[r_key]["relation"]["RELATED"] == 1:
                    if ent_key[0] == r_key[0] and ent_key[1] == r_key[1]:
                        t_after += relations[r_key]["tuple"][1] + " "
                    elif ent_key[0] == r_key[2] and ent_key[1] == r_key[3]:
                        t_before += relations[r_key]["tuple"][0] + " "

            classification_string = (f"{t_before}{entity} {t_after}").strip()
            print(f"{classification_string} | {label}")
