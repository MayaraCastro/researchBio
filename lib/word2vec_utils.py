from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors

model = KeyedVectors.load_word2vec_format("/content/drive/My Drive/bio_files/PubMed-shuffle-win-30.bin", binary=True)


def words_to_vector(words):
  try:
    vector = sum(model.get_vector(w.text) for w in words)/len(words)
  except Exception as e:
    #print(e, type(words), words)
    pass
  return vector


def get_cosine_similarity(term, entity):
    cosine_similarities = []
    try:
        cosine_similarities.append(model.cosine_similarities(words_to_vector(entity[0]), [words_to_vector(term[0])])[0])
    except Exception as e:
        pass
    for synonym in term[1]['synonyms']:
        try:
            cosine_similarities.append(
                model.cosine_similarities(words_to_vector(entity[0]), [words_to_vector(synonym)])[0])
        except Exception as e:
            pass
    return max(cosine_similarities)