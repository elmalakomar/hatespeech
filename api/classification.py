import pandas as pd
from api.functions import *
from api.setting import Settings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import demoji
import emot
from autocorrect import Speller
from joblib import Parallel, delayed

import multiprocessing

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import re

def run(settings):
    data = pd.read_csv(settings.csv_path)
    data.drop(settings.ignore_columns, axis=1, inplace=True)
    X = data['text']
    print(type(X))

    Y = data['annotation']
    Y = Y.map({'none': 0, 'racism': 1, 'sexism': 1})
    ## -- Preprocessing --
    #X = X.apply(lambda x: clean_tweet(x))
    X = pd.Series(Parallel(n_jobs=-1, verbose=5, backend="multiprocessing")(delayed(clean_tweet)(x) for x in X))
    print(type(X))
    if settings.LOWER_TEXT:
        X = X.str.lower()

    if settings.REMOVE_NUMBERS:
        X = X.str.replace("[0-9]", " ")

    if settings.AUTOCORRECT: #TODO: DA SISTEMARE
        print("autocorrect")
        spell = Speller(fast=True)
        #X = X.apply(lambda x: spell(x))
        X = Parallel(n_jobs=-1, verbose=10, backend="multiprocessing")(delayed(spell)(x) for x in X)
        print("ended")

    frequent_dict = most_frequent_words(X)
    print(len(frequent_dict)) # 31953 distinct words
    #
    # #stopword_set = set(stopwords.words('english'))
    # #print(len(stopword_set))
    # with open('../data/mystopwords.txt', 'w') as f:
    #     for item in frequent_dict:
    #         f.write("%s\n" % item[0])

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    print(vectorizer.stop_words_)
    print("Input X after preprocessing: {}".format(X.shape))

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

    # apply ML on the data
    model = MultinomialNB()
    model.fit(Xtrain, Ytrain)
    pred = model.predict(Xtest)

    # evaluate
    print("Classification rate for NB:", model.score(Xtest, Ytest))

    print("confusion matrix:")
    print(metrics.confusion_matrix(Ytest, pred))

    print("classification report:")
    print(metrics.classification_report(Ytest, pred, target_names=["none", "hate"]))


