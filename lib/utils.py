import spacy
import string
import csv
from pronto import Ontology
onto_path = '/content/drive/My Drive/bio_files/OntoBiotope_BioNLP-OST-2019 (1).obo'
onto = Ontology(onto_path)


def remove_stopwords(sentence):
    non_stop_words = [word.text for word in sentence if not word.is_stop]
    return nlp(' '.join(non_stop_words))


def remove_non_ascii(text):
    return nlp(' '.join([token.text for token in text if all([letter in string.ascii_letters for letter in token.text])]))


def load_entities(entities_loc):
    names = dict()
    descriptions = dict()
    test = []
    entities = []
    with open(entities_loc, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter="\t")

        for row in csvreader:
            qid = row[0]
            name = row[1]
            desc = row[2]
            names[qid] = name
            descriptions[qid] = desc
            try:
                test.append((remove_non_ascii(remove_stopwords(nlp(desc))),
                             {"start": int(name.split(' ')[1]),
                              "end": int(name.split(' ')[2]),
                              "type": name.split(' ')[0],
                              "qid": qid}))
            except:
                pass
    return test  # names, descriptions


def load_concepts_ontology():
    entities = []
    for term in onto.terms():
        synonyms = []
        for synonym in term.synonyms:
            synonyms.append(remove_non_ascii(remove_stopwords(nlp(synonym.description))))

        entities.append(
            (remove_non_ascii(remove_stopwords(nlp(term.name))),
             {"synonyms": synonyms,
              "concept": term})
        )

    return entities

noun_tags_list = ["NOUN", "PROPN"]
def get_most_informative_word(doc):
  for noun_phrase in doc.noun_chunks:
      for token in noun_phrase.as_doc():
        if token.pos_ in noun_tags_list:
          return token
