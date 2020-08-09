import pandas as pd
from autocorrect import Speller
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import EditedNearestNeighbours

from api.functions import *
from api.preprocessing import *

def run(settings):

    _,X,Y = csv2data(settings.CSV_PATH,settings.X_LABEL,settings.Y_LABEL, settings.IGNORE_COLUMNS, settings.BALANCE_DATA)

    # encoding labels: binary classification since there are little occurrences of 'racism' samples
    Y = Y.map({'none': 0, 'racism': 1, 'sexism': 1})

    ## -- Preprocessing --
    if settings.CLEAN_TEXT:
        X = pd.Series(Parallel(n_jobs=-1, verbose=5, backend="multiprocessing")(delayed(clean_tweet)(settings,x) for x in X))

    if settings.AUTOCORRECT:
        print("autocorrect")
        spell = Speller(fast=True)
        #X = X.apply(lambda x: spell(x))
        X = Parallel(n_jobs=-1, verbose=10, backend="multiprocessing")(delayed(spell)(x) for x in X)
        print("ended")

    ## -- data analysis and visualization
    if settings.MOST_FREQUENT_WORDS:
        dic = get_frequent_words(X)
        visualize_frequent_words(dic,20)

    if settings.WORDCLOUD:
        worldCloud(X,Y)

    ## -- feature vector
    vectorizer = CountVectorizer()
    X_fv = vectorizer.fit_transform(X)
    #print("Input X after preprocessing: {}".format(X_fv.shape))

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=settings.RANDOM_SEED)

    for train_index, test_index in skf.split(X_fv,Y):
        Xtrain, Xtest = X_fv[train_index], X_fv[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]

        if settings.UNDERSAMPLING:
            enn = EditedNearestNeighbours(n_neighbors=3,n_jobs=4)
            Xtrain, Ytrain = enn.fit_resample(Xtrain,Ytrain)
        # -- classification
        print(Xtrain.shape, Xtest.shape)
        print(Ytrain.shape, Ytest.shape)

        model = LogisticRegression()
        model.fit(Xtrain, Ytrain)
        pred = model.predict(Xtest)
        print(metrics.confusion_matrix(Ytest, pred))
        print(metrics.classification_report(Ytest, pred, target_names=["none", "hate"]))



    # Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_fv, Y, test_size=0.33)
    # # -- classification
    # print(Xtest.shape)
    # print(Ytest.shape)
    # print(Ytest[:1])
    # print(Ytest[:1].index)
    # # apply ML on the data
    # model = LogisticRegression()
    # model.fit(Xtrain, Ytrain)
    # pred = model.predict(Xtest)
    #
    # #get index of error prediction
    # false_positives = []
    # false_negatives = []
    # indices = Ytest.index
    # for i,y_test in enumerate(Ytest):
    #     if y_test != pred[i] and pred[i] == 0:
    #         false_positives.append(indices[i])
    #     if y_test != pred[i] and pred[i] == 1:
    #         false_negatives.append(indices[i])
    #
    # print("--->", len(false_positives))
    # print("--->", len(false_negatives))
    #
    # # evaluate
    # print("Classification rate:", model.score(Xtest, Ytest))
    #
    # print("confusion matrix:")
    # print(metrics.confusion_matrix(Ytest, pred))
    #
    # print("classification report:")
    # print(metrics.classification_report(Ytest, pred, target_names=["none", "hate"]))
    #
    # # analyze wrong prediction
    # X_FP = X.iloc[false_positives]
    # print(X_FP.shape)
    # print(X_FP[:2])
    # dicFP = get_frequent_words(X_FP)
    # visualize_frequent_words(dicFP, 30,title='False Positive')
    #
    # X_FN = X.iloc[false_negatives]
    # print(X_FN.shape)
    # print(X_FN[:2])
    # dicFN = get_frequent_words(X_FN)
    # visualize_frequent_words(dicFN, 30, title='False Negative')
    #
    #
    #
    #
    #
