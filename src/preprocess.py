import os
import re
import pandas as pd 
import numpy as np 
import distance
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import config
import warnings
warnings.filterwarnings("ignore")
nltk.download('stopwords')


class DataPreprocess:

    def __init__(self, data, SAFE_DIV = 0.0001):
        self.data = data
        self.SAFE_DIV = SAFE_DIV 
        self.STOP_WORDS = stopwords.words('english')
        
    @staticmethod
    def question_preprocess(x):
        x = str(x).lower()
        x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
            .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
            .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
            .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
            .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
            .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
            .replace("€", " euro ").replace("'ll", " will").replace("し","").replace("シ","")

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
    
    # @staticmethod
    def get_token_features(self, q1, q2):
        token_features = [0.0]*10
        
        # Converting the Sentence into Tokens: 
        q1_tokens = q1.split()
        q2_tokens = q2.split()

        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return token_features
        # Get the non-stopwords in Questions
        q1_words = set([word for word in q1_tokens if word not in self.STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in self.STOP_WORDS])
        
        #Get the stopwords in Questions
        q1_stops = set([word for word in q1_tokens if word in self.STOP_WORDS])
        q2_stops = set([word for word in q2_tokens if word in self.STOP_WORDS])
        
        # Get the common non-stopwords from Question pair
        common_word_count = len(q1_words.intersection(q2_words))
        
        # Get the common stopwords from Question pair
        common_stop_count = len(q1_stops.intersection(q2_stops))
        
        # Get the common Tokens from Question pair
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
        
        
        token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + self.SAFE_DIV)
        token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + self.SAFE_DIV)
        token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + self.SAFE_DIV)
        token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + self.SAFE_DIV)
        token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + self.SAFE_DIV)
        token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + self.SAFE_DIV)
        
        # Last word of both question is same or not
        token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
        
        # First word of both question is same or not
        token_features[7] = int(q1_tokens[0] == q2_tokens[0])
        
        token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
        
        #Average Token Length of both Questions
        token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
        return token_features
    
    @staticmethod
    def normalized_word_Common(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * len(w1 & w2)

    @staticmethod
    def normalized_word_share(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))

    @staticmethod
    def get_longest_substr_ratio(a, b):
        strs = list(distance.lcsubstrings(a, b))
        if len(strs) == 0:
            return 0
        else:
            return len(strs[0]) / (min(len(a), len(b)) + 1)

    def create_new_features(self):
        # normal custom features

        self.data["question1"] = self.data["question1"].fillna("").apply(self.question_preprocess)
        self.data["question2"] = self.data["question2"].fillna("").apply(self.question_preprocess)

        self.data['freq_qid1'] = self.data.groupby(['qid1'])['qid1'].transform('count') 
        self.data['freq_qid2'] = self.data.groupby('qid2')['qid2'].transform('count')

        self.data['q1len'] = self.data['question1'].str.len()
        self.data['q2len'] = self.data['question1'].str.len()

        self.data['q1_n_words'] = self.data['question1'].apply(lambda row: len(row.split(" ")))
        self.data['q2_n_words'] = self.data['question2'].apply(lambda row: len(row.split(" ")))

        self.data['word_Common'] = self.data.apply(self.normalized_word_Common, axis=1)
        self.data['word_share'] = self.data.apply(self.normalized_word_share, axis=1)

        self.data['word_Total'] = self.data['q1_n_words'] + self.data['q2_n_words']

        self.data['freq_q1+q2'] = self.data['freq_qid1'] + self.data['freq_qid2']
        self.data['freq_q1-q2'] = abs(self.data['freq_qid1'] - self.data['freq_qid2'])

        # advanced features
        self.token_features = self.data.apply(lambda x: self.get_token_features(x["question1"], x["question2"]), axis=1)

        self.data["cwc_min"] = list(map(lambda x: x[0], self.token_features))
        self.data["cwc_max"] = list(map(lambda x: x[1], self.token_features))
        self.data["csc_min"] = list(map(lambda x: x[2], self.token_features))
        self.data["csc_max"] = list(map(lambda x: x[3], self.token_features))
        self.data["ctc_min"] = list(map(lambda x: x[4], self.token_features))
        self.data["ctc_max"] = list(map(lambda x: x[5], self.token_features))
        self.data["last_word_eq"] = list(map(lambda x: x[6], self.token_features))
        self.data["first_word_eq"] = list(map(lambda x: x[7], self.token_features))
        self.data["abs_len_diff"] = list(map(lambda x: x[8], self.token_features))
        self.data["mean_len"] = list(map(lambda x: x[9], self.token_features))

        self.data["token_set_ratio"]       = self.data.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
        self.data["token_sort_ratio"]      = self.data.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
        self.data["fuzz_ratio"]            = self.data.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
        self.data["fuzz_partial_ratio"]    = self.data.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)

        self.data["longest_substr_ratio"]  = self.data.apply(lambda x: self.get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)

        return self.data

