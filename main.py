import os

import spacy
import string
import csv
from spacy.lang.en import English

from lib import utils
from lib import word2vec_utils

nlp = spacy.load("en_core_web_sm")

from spacy.tokens import Doc

DATASET_DIR_PATH = os.path.join(os.path.dirname(__file__), 'resources/dataset/training')

nlp.vocab["and"].is_stop = False

files = os.listdir(DATASET_DIR_PATH)

annotated_entities = {}
for file in files:
    if ".a1" in file:
        annotated_entities.update({file: utils.load_entities(os.path.join(DATASET_DIR_PATH, file))})

ontology_concepts = utils.load_concepts_ontology()

k = 10
for entities in annotated_entities.values():
    for entity in entities:
        entity_concept_similarity = []
        for concept in ontology_concepts:
            try:
                entity_concept_similarity.append(
                    {'concept': concept,
                     'cosine_similarity': word2vec_utils.get_cosine_similarity(concept, entity)})
            except:
                pass

        entity_concept_similarity = sorted(entity_concept_similarity,
                                           key=lambda x: x['cosine_similarity'],
                                           reverse=True)
        entity[1].update({'most_similar_concepts': entity_concept_similarity[:k]})

for entities in annotated_entities.values():
    for entity in entities:
        entity[1].update({'most_informative_word': utils.get_most_informative_word(entity[0])})
