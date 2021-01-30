import streamlit as st
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


# changing page main title and main icon(logo)
PAGE_CONFIG = {"page_title":"Question's Similarity", "page_icon":":book:", "layout":"centered"}
st.set_page_config(**PAGE_CONFIG)   

st.sidebar.text("Created on Sat, Jan 18 2021")
st.sidebar.markdown("**@author:Sumit Kumar** :monkey_face:")
st.sidebar.markdown("[My Github](https://github.com/IMsumitkumar) :penguin:")
st.sidebar.markdown("[findingdata.ml](https://www.findingdata.ml/) :spider_web:")
st.sidebar.markdown("[Data & Description](https://www.kaggle.com/c/quora-question-pairs/data) :house:")
st.sidebar.markdown("coded with :heart:")

# sidebar header
st.sidebar.subheader("Qustion's pair similarity")

# sidebar : choose analysis or prediction page
option = st.sidebar.selectbox(
    'prediction? Select From here...',
     ("Please Select here", "Check similiar questions"))

@st.cache
def processing_question(x):
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

def normalized_word_Common(question1, question2):
    w1 = set(map(lambda word: word.lower().strip(), question1[0].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), question2[0].split(" ")))    
    return 1.0 * len(w1 & w2)

def normalized_word_share(question1, question2):
    w1 = set(map(lambda word: word.lower().strip(), question1[0].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), question2[0].split(" ")))    
    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))


@st.cache
def load_sample_data():
    test_data = pd.read_csv("input/sample_data.csv")
    return test_data
    
if option == "Please Select here":
    st.title("Qustion's pair similarity")
    st.text("Can you identify question pairs that have the same intent?")
    st.markdown("Data is taken from Quora, which is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.")
    # st.title("")
    st.image("https://i.imgur.com/EcyN4up.jpg", width=650)
    st.markdown("Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.")
    st.markdown("I have used Logistic Regression model to identify duplicate questions with a 0.40 log loss.")
    st.markdown("The goal of this project is to predict which of the provided pairs of questions contain two questions with the same meaning. The ground truth is the set of labels that have been supplied by human experts. The ground truth labels are inherently subjective, as the true meaning of sentences can never be known with certainty.")

elif option == "Check similiar questions":
    if st.checkbox("Use Sample Questions"):
        sample_data = load_sample_data()
        k = np.random.randint(0, sample_data.shape[0]-1)
        ques1 = sample_data['question1'][k]
        ques2 = sample_data['question2'][k]
        st.image("https://i.imgur.com/2SrlkoH.jpg", width=700)
        first_text = [st.text_area("Question1 goes here...", ques1, height=100)]
        second_text = [st.text_area("Question2 goes here...", ques2, height=100)]

    else:
        st.image("https://i.imgur.com/2SrlkoH.jpg", width=700)
        first_text = [st.text_area("Question1 goes here...", height=100)]
        second_text = [st.text_area("Question2 goes here...", height=100)]

    
    q1_vect = pickle.load(open('models/q1_vect.pkl', 'rb'))
    q2_vect = pickle.load(open('models/q2_vect.pkl', 'rb'))
    model = pickle.load(open('models/lgr_model.pkl', 'rb')) 

    

    if st.button("Predict"):

        question1 = [processing_question(first_text)]
        q1_len = len(question1)
        q1_n_words = len(question1[0].split(' '))

        question2 = [processing_question(second_text)]
        q2_len = len(question2)
        q2_n_words = len(question2[0].split(' '))

        word_share = normalized_word_share(question1, question2)
        word_Common = normalized_word_Common(question1, question2)

        que1_vector = q1_vect.transform(list(question1))
        que2_vector = q2_vect.transform(list(question2))

        new_feature = [q1_len, q2_len, q1_n_words, q2_n_words, word_share, word_Common]
        query_vector = hstack((que1_vector, que2_vector, new_feature)).tocsr()

        result = model.predict(query_vector)
        result_prob = model.predict_proba(query_vector)
        
        st.success(result_prob[0])

        if result[0] == 0:
            st.success("Not Similiar")
        else:
            st.success("Similiar")