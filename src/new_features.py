import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from sklearn import preprocessing
import pickle
from scipy.sparse import hstack

def preprocess(x):
  x = str(x).lower()
  x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
        .replace("€", " euro ").replace("'ll", " will")

  x = re.sub(r"([0-9]+)000000", r"\1m", x)
  x = re.sub(r"([0-9]+)000", r"\1k", x)
  porter = PorterStemmer()
  pattern = re.compile('\W')

  if type(x) == type(''):
    x  =re.sub(pattern, ' ', x)

  if type(x) == type(' '):
    x = porter.stem(x)
    example = BeautifulSoup(x, "html.parser")
    x = example.get_text()

  return x

q1_vect = pickle.load(open('../models/q1_vect.pkl', 'rb'))
q2_vect = pickle.load(open('../models/q2_vect.pkl', 'rb'))
model = pickle.load(open('../models/lgr_model.pkl', 'rb')) 

question1 = ["how can i increase the speed of my internet connection while using a vpn"]
question2 = ["how can internet speed be increased by hacking through dns"]

question1 = preprocess(question1)
question2 = preprocess(question2)

que1_vector = q1_vect.transform(list(question1))
que2_vector = q2_vect.transform(list(question2))

q1_len = len(question1)
q2_len = len(question2)

q1_n_words = len(question1.split(' '))
q2_n_words = len(question2.split(' '))

def normalized_word_Common(question1, question2):
  w1 = set(map(lambda word: word.lower().strip(), question1.split(" ")))
  w2 = set(map(lambda word: word.lower().strip(), question2.split(" ")))    
  return 1.0 * len(w1 & w2)

def normalized_word_share(question1, question2):
  w1 = set(map(lambda word: word.lower().strip(), question1.split(" ")))
  w2 = set(map(lambda word: word.lower().strip(), question2.split(" ")))    
  return 1.0 * len(w1 & w2)/(len(w1) + len(w2))


word_share = normalized_word_share(question1, question2)
word_Common = normalized_word_Common(question1, question2)

new_feature = [q1_len, q2_len, q1_n_words, q2_n_words, word_share, word_Common]

from scipy.sparse import hstack
query_vector = hstack((que1_vector, que2_vector, new_feature)).tocsr()

print(model.predict_proba(query_vector))