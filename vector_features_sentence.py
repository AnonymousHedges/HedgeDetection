# File for feature processing
# For that, we will compute what are called n-gram, and see their association with. 
# For example, the 2-grams derived from the sentence "I am listening to music" are [["I", "am"], ["am", "listening"], ["listening", "to"], ["to", "music"]]
# We will focus on n = 1, 2 and 3.

from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
import numpy as np
import liwc
import re
import spacy
from spacy.lang.en.examples import sentences 

# Remember to type : python3 -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")

nltk.download("punkt")
nltk.download('wordnet')
parse, category_names = liwc.load_token_parser('LIWC2015.dic')  

PATH_POS_DATASET = "pos_dataset.txt"

ALPHA = 0.05

def get_liwc(sent):
    tokens = word_tokenize(sent.lower())
    # The reason why we use that instead of a Counter is that we want to keep a strict order for the categories in the sentence.
    counts = {k:0 for k in category_names}
    for token in tokens:
        list_cat = parse(token)
        for cat in list_cat:
            counts[cat] += 1

    return list(counts.values())

def get_ngram(sentences, train = False, runtime_dict = None, threshold=50):
    """
    In : sentences : List(List(Str)) : the list of conversations.
         train : Bool : Whether or not to construct the n-gram dictionnaries.
         runtime_dict : List(Dict) : the list of dictionnaries for the 1-gram, 2-gram and 3-gram.
         threshold : int : minimum number of occurences in the training dataset before we count one n-gram as part of the vocabulary.
    """

    # train and runtime_dict should not both be empty
    assert train or runtime_dict

    lemmatizer = WordNetLemmatizer()
 
    # Here, we use a lemmatizer, transforming a word into its lemma (morphologic root) -> "thought" becomes "think", "walking" becomes "walk"
    sentences_unigram = [[lemmatizer.lemmatize(word) for word in word_tokenize(sent)] for sent in sentences]
    sentences_bigram = [[(sent[i], sent[i+1]) for i in range(len(sent)-1)] for sent in sentences_unigram]

    if train:
      dict_ngram = {1:defaultdict(int), 2:defaultdict(int)}

      for sent_unigram in sentences_unigram:
          for unigram in sent_unigram:
              dict_ngram[1][unigram] += 1

      for sent_bigram in sentences_bigram:
          for bigram in sent_bigram:
              dict_ngram[2][bigram] += 1

      runtime_dict = {1:defaultdict(int), 2:defaultdict(int)}
      # We do a first pass in the dictionnary to keep only the words that occur at least #threshold# time in the training dataset.
      for n, v_ngram in dict_ngram.items():
          for ngram, occurence_count in v_ngram.items():
            if occurence_count >= threshold:
              runtime_dict[n][ngram] = occurence_count

    vectors_ngram = []

    for index_sent in range(len(sentences)):
      dict_to_vectors = {1:{k:0 for k,v in runtime_dict[1].items()}, 2:{k:0 for k,v in runtime_dict[2].items()}}

      for unigram in sentences_unigram[index_sent]:
          if unigram in dict_to_vectors[1]:
            dict_to_vectors[1][unigram] += 1

      for bigram in sentences_bigram[index_sent]:
          if bigram in dict_to_vectors[2]:
            dict_to_vectors[2][bigram] += 1

      vectors_ngram.append(list(dict_to_vectors[1].values()) + list(dict_to_vectors[2].values()))

    if train:
      return runtime_dict, vectors_ngram
    else:
      return vectors_ngram

def get_ngram_pos_tag(sentences, train = False, runtime_dict = None, threshold=10):
    """
    In : sentences : List(List(Str)) : the list of conversations.
         train : Bool : Whether or not to construct the n-gram dictionnaries.
         runtime_dict : List(Dict) : the list of dictionnaries for the 1-gram, 2-gram and 3-gram.
         threshold : int : minimum number of occurences in the training dataset before we count one n-gram as part of the vocabulary.
    """

    # train and runtime_dict should not both be empty
    assert train or runtime_dict

    # Instead of using a tokenizer and a lemmatizer, we are going to use the POS-tagger provided in spaCy.
    # Apart from that line, the function is pretty similar to the previous one.
    
    sentences_unigram = [[w.pos_ for w in nlp(sent)] for sent in sentences]
    sentences_bigram = [[(sent[i], sent[i+1]) for i in range(len(sent)-1)] for sent in sentences_unigram]
    sentences_trigram = [[(sent[i], sent[i+1], sent[i+2]) for i in range(len(sent)-2)] for sent in sentences_unigram]

    dict_ngram = {1:defaultdict(int), 2:defaultdict(int), 3:defaultdict(int)}
    if train:
        for sent_unigram in sentences_unigram:
            for unigram in sent_unigram:
                dict_ngram[1][unigram] += 1

        for sent_bigram in sentences_bigram:
            for bigram in sent_bigram:
                dict_ngram[2][bigram] += 1

        for sent_trigram in sentences_trigram:
            for trigram in sent_trigram:
                dict_ngram[3][trigram] += 1

        runtime_dict = {1:defaultdict(int), 2:defaultdict(int), 3:defaultdict(int)}
        # We do a first pass in the dictionnary to keep only the words that occur at least #threshold# time in the training dataset.
        for n, v_ngram in dict_ngram.items():
            for ngram, occurence_count in v_ngram.items():
              if occurence_count >= threshold:
                runtime_dict[n][ngram] = occurence_count

    vectors_ngram = []

    for index_sent in range(len(sentences)):
      dict_to_vectors = {1:{k:0 for k,v in runtime_dict[1].items()}, 2:{k:0 for k,v in runtime_dict[2].items()}, 3:{k:0 for k,v in runtime_dict[3].items()}}

      for unigram in sentences_unigram[index_sent]:
          if unigram in dict_to_vectors[1]:
            dict_to_vectors[1][unigram] += 1

      for bigram in sentences_bigram[index_sent]:
          if bigram in dict_to_vectors[2]:
            dict_to_vectors[2][bigram] += 1

      for trigram in sentences_trigram[index_sent]:
          if trigram in dict_to_vectors[3]:
            dict_to_vectors[3][trigram] += 1

      vectors_ngram.append(list(dict_to_vectors[1].values()) + list(dict_to_vectors[2].values()) + list(dict_to_vectors[3].values()))

    if train:
      return runtime_dict, vectors_ngram
    else:
      return vectors_ngram


