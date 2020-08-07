import pandas as pd
from autocorrect import Speller
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from api.functions import *
from api.preprocessing import *

def run(settings):

    _,X,Y = csv2data(settings.CSV_PATH,settings.X_LABEL,settings.Y_LABEL, settings.IGNORE_COLUMNS)

    # encoding labels: binary classification since there are little occurrences of 'racism' samples
    Y = Y.map({'none': 0, 'racism': 1, 'sexism': 1})
    ## -- Preprocessing --
    if settings.CLEAN_TEXT:
        X = pd.Series(Parallel(n_jobs=-1, verbose=5, backend="multiprocessing")(delayed(clean_tweet)(settings,x) for x in X))

    if settings.LOWER_TEXT:
        X = X.str.lower()

    if settings.AUTOCORRECT:
        print("autocorrect")
        spell = Speller(fast=True)
        #X = X.apply(lambda x: spell(x))
        X = Parallel(n_jobs=-1, verbose=10, backend="multiprocessing")(delayed(spell)(x) for x in X)
        print("ended")

    dic = get_frequent_words(X)
    visualize_frequent_words(dic,20)
    # frequent_dict = most_frequent_words(X)
    # print(len(frequent_dict)) # 31953 distinct words

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    print("Input X after preprocessing: {}".format(X.shape))

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

    # apply ML on the data
    model = LogisticRegression()
    model.fit(Xtrain, Ytrain)
    pred = model.predict(Xtest)

    # evaluate
    print("Classification rate:", model.score(Xtest, Ytest))

    print("confusion matrix:")
    print(metrics.confusion_matrix(Ytest, pred))

    print("classification report:")
    print(metrics.classification_report(Ytest, pred, target_names=["none", "hate"]))



