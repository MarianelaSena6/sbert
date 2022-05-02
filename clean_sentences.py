import nltk
import pandas as pd
import re
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('portuguese'))


# Elimina las palabras que no son importantes (stopwords y palabras de 2 letras o menos)
def remove_stopwords_br(sentence):
    sentence = [token.lower() for token in nltk.word_tokenize(sentence) if (token.lower() not in stop_words)]
    return " ".join(sentence)


def remove_shortwords(sentence):
    sentence = [token for token in nltk.word_tokenize(sentence) if len(token) > 2]
    return " ".join(sentence)


def remove_accents(sent):
    trans = str.maketrans('àáâãäåèéêëìíîïòóôõöùúûü', 'aaaaaaeeeeiiiiooooouuuu')
    sent = sent.translate(trans)
    return sent


def clean(sentence):
    sentence = re.sub(r'\d', ' ', sentence)
    sentence = sentence.replace('tv', 'televisao')
    sentence = sentence.replace('ñ', 'n')
    sentence = sentence.replace('\x81', '')
    sentence = sentence.replace('ç', 'c')
    # sentence = sentence.translate(str.maketrans(' ', ' ', string.punctuation))
    sentence = re.sub(r"[!\"'#$%&()*+,-./:;<=>?@[\\\]^_`{|}~]", ' ', sentence)
    sentence = re.sub(r'\s{2,}', ' ', sentence)
    sentence = remove_accents(sentence)
    sentence = remove_shortwords(sentence)
    sentence = sentence.strip()
    return sentence


def create_vocab(doc_list):
    vocab = {}
    for doc in doc_list:
        doc = doc.translate(str.maketrans('', '', string.punctuation))

        words = nltk.word_tokenize(doc.lower())
        for word in words:
            if word in vocab.keys():
                vocab[word] = vocab[word] + 1
            else:
                vocab[word] = 1
    return vocab



