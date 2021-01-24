import joblib
import argparse
import os
import pandas as pd 
import numpy as np 
from sklearn import metrics
from sklearn import linear_model
from sklearn import tree
import model_dispatcher
import config
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from preprocess import DataPreprocess
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

def run(data, vectorizer, model):

    y = data['is_duplicate']
    X = data.drop(columns=['is_duplicate'], axis=1)

    vect1 = model_dispatcher.vectorizers[vectorizer]
    vect1.fit_transform(X['question1'])
    quest1_vec = vect1.transform(X['question1'].values)

    vect2 = model_dispatcher.vectorizers[vectorizer]
    vect2.fit_transform(X['question1'])
    quest2_vec = vect2.transform(X['question1'].values)

    X.drop(columns=['id', 'qid1', 'qid2', 'question1', 'question2'], axis=1, inplace=True)

    final_X = hstack((X, quest1_vec, quest2_vec)).tocsr()


    X_train, X_test, y_train, y_test = train_test_split(final_X, y, stratify=y, test_size=0.3)

    clf = model_dispatcher.models[model]

    clf.fit(X_train, y_train)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(X_train, y_train)
    predict_y = sig_clf.predict_proba(X_train)
    print("Log loss for tfidf train data : ",log_loss(y_train, predict_y,  eps=1e-15))
    predict_y = sig_clf.predict_proba(X_test)
    print("Log loss for tfidf test data : ",log_loss(y_test, predict_y,  eps=1e-15))

    joblib.dump(
        clf, 
        os.path.join(config.MODEL_OUTPUT, f"{model}_{vectorizer}.bin")
    )
    print("Model Saved!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vectorizer",
        type=str
    )

    parser.add_argument(
        "--model",
        type=str
    )

    args = parser.parse_args()

    training_file = config.TRAINING_FILE
    data = pd.read_csv(training_file)
    data = data[0:100000]

    preprocess_data = DataPreprocess(data)
    data = preprocess_data.create_new_features()
  
    run(
        data = data,        
        vectorizer=args.vectorizer,
        model=args.model
    )


