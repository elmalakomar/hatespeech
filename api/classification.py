import numpy as np
import pandas as pd
from autocorrect import Speller
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.utils import shuffle

from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

from api.functions import *
from api.preprocessing import *
from pprint import pprint
import os
from sklearn.pipeline import FeatureUnion

def run(settings):
    #pprint(settings.__dict__)

    data,X,Y = csv2data(settings.CSV_PATH,
                     settings.X_LABEL,
                     settings.Y_LABEL,
                     ignore_columns = settings.IGNORE_COLUMNS,
                     balance_data = settings.BALANCE_DATA,
                     random = settings.RANDOM_SEED)
    Y = Y.map({'none': 0, 'racism': 1, 'sexism': 1, 'N': 0, 'H': 1})
    #print(Y.value_counts(normalize=True).values)
    if settings.PRINT_PIE:
        index = (Y.value_counts(normalize=True).index).map({0:'none', 1: 'hate'})
        x = Y.value_counts(normalize=True).values
        plt.pie(x,labels=index,autopct='%1.1f%%',textprops={'fontsize': 15},startangle=90)
        plt.savefig("../plots/pie_not_balanced.pdf", bbox_inches = 'tight')
        plt.show()

    if settings.ADD_DATA:
        data2,_,_ = csv2data('../data/onlineHarassment.csv',
                         settings.X_LABEL,
                         settings.Y_LABEL,
                         random=settings.RANDOM_SEED)

        hate_data = pd.DataFrame([row[1] for row in data2.iterrows() if row[1][settings.Y_LABEL] == 'H'])
        data = pd.concat([data,hate_data], ignore_index=True)
        X = data[settings.X_LABEL]
        Y = data[settings.Y_LABEL]

        # encoding labels: binary classification since there are little occurrences of 'racism' samples
        Y = Y.map({'none': 0, 'racism': 1, 'sexism': 1, 'N':0, 'H':1})
        if settings.PRINT_PIE:
            index = (Y.value_counts(normalize=True).index).map({0: 'none', 1: 'hate'})
            x = Y.value_counts(normalize=True).values
            plt.pie(x, labels=index, autopct='%1.1f%%', textprops={'fontsize': 15},startangle=90)
            plt.savefig("../plots/pie_balanced.pdf", bbox_inches='tight')
            plt.show()
    ## -- Preprocessing --
    if settings.CLEAN_TEXT:
        X = pd.Series(Parallel(n_jobs=-1, verbose=5, backend="multiprocessing")(delayed(clean_tweet)(settings,x) for x in X))

    # if settings.AUTOCORRECT:
    #     print("autocorrect")
    #     spell = Speller(fast=True)
    #     #X = X.apply(lambda x: spell(x))
    #     X = Parallel(n_jobs=-1, verbose=10, backend="multiprocessing")(delayed(spell)(x) for x in X)
    #     print("ended")

    ## -- data analysis and visualization
    if settings.MOST_FREQUENT_WORDS:
        dic = get_frequent_words(X)
        visualize_frequent_words(dic,20,save_fig=True)

    if settings.WORDCLOUD:
        worldCloud(X,Y,settings.X_LABEL,settings.Y_LABEL,label=1)
        worldCloud(X, Y, settings.X_LABEL, settings.Y_LABEL, label=0)

    ## -- feature vector
    #vectorizer = FeatureUnion([("count", CountVectorizer()), ("tf-idf", TfidfVectorizer())])
    #vectorizer = TfidfVectorizer()
    vectorizer = CountVectorizer()
    X_fv = vectorizer.fit_transform(X)
    #print("Input X after preprocessing: {}".format(X_fv.shape))

    #Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_fv, Y, test_size=0.20, random_state=settings.RANDOM_SEED)
    precision_N = 0
    recall_N = 0
    f1_N = 0
    support_N = 0

    precision_h = 0
    recall_h = 0
    f1_h = 0
    support_h = 0

    false_positives = []
    false_negatives = []

    entropyequal = defaultdict(int)
    entropy_notequal = defaultdict(int)

    n_splits = 5
    cm = 0
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=settings.RANDOM_SEED)
    for train_index, test_index in skf.split(X_fv,Y):
        Xtrain, Xtest = X_fv[train_index], X_fv[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]


        if settings.UNDERSAMPLING:
            print(Ytrain.shape)
            enn = EditedNearestNeighbours(n_neighbors=3,n_jobs=4)
            Xtrain, Ytrain = enn.fit_resample(Xtrain,Ytrain)
            print(Ytrain.shape)

        if settings.OVERSAMPLING:
            print(Ytrain.shape)
            smote = SMOTE(k_neighbors=3,n_jobs=4)
            Xtrain, Ytrain = smote.fit_resample(Xtrain, Ytrain)
            print(Ytrain.shape)

        # -- classification
        #print(Xtrain.shape, Xtest.shape)
        #print(Ytrain.shape, Ytest.shape)

        model = LogisticRegression(solver='liblinear',max_iter=100, n_jobs=-1)
        model.fit(Xtrain, Ytrain)
        pred = model.predict(Xtest)

        if settings.ENTROPY:

            pred_proba = model.predict_proba(Xtest)
            print(Ytest.shape,pred_proba.shape)
            print(Ytest.iloc[0], np.argmax(pred_proba[0]), pred_proba[0])
            for index,prob in enumerate(pred_proba):
                temp = max(prob) * 10
                if Ytest.iloc[index] == np.argmax(prob) :
                    if 5 <= temp < 6:
                        entropyequal[5] += 1
                    if 6 <= temp < 7:
                        entropyequal[6] += 1
                    if 7 <= temp < 8:
                        entropyequal[7] += 1
                    if 8 <= temp < 9:
                        entropyequal[8] += 1
                    if 9 <= temp <= 10:
                        entropyequal[9] += 1
                elif Ytest.iloc[index] != np.argmax(prob):
                    if 5 <= temp < 6:
                        entropy_notequal[5] += 1
                    if 6 <= temp < 7:
                        entropy_notequal[6] += 1
                    if 7 <= temp < 8:
                        entropy_notequal[7] += 1
                    if 8 <= temp < 9:
                        entropy_notequal[8] += 1
                    if 9 <= temp <= 10:
                        entropy_notequal[9] += 1

        cm_temp = metrics.confusion_matrix(Ytest, pred)
        print(cm_temp)
        cm += cm_temp
        print(metrics.classification_report(Ytest, pred, target_names=["none", "hate"]))
        report = metrics.precision_recall_fscore_support(Ytest, pred,beta=1,average=None, labels=[0, 1])
        precision_N += report[0][0]*report[3][0]
        recall_N += report[1][0] * report[3][0]
        f1_N += report[2][0]*report[3][0]
        support_N += report[3][0]

        precision_h +=  report[0][1]*report[3][1]
        recall_h += report[1][1] * report[3][1]
        f1_h += report[2][1]*report[3][1]
        support_h += report[3][1]

        if settings.EVALUATE_FALSE:
            #get index of error prediction
            # false_positives = []
            # false_negatives = []
            indices = Ytest.index
            for i,y_test in enumerate(Ytest):
                if y_test != pred[i] and pred[i] == 0:
                    false_positives.append(indices[i])
                if y_test != pred[i] and pred[i] == 1:
                    false_negatives.append(indices[i])
            #
            print("--->", len(false_positives))
            print("--->", len(false_negatives))

            # analyze wrong prediction
            # X_FP = X.iloc[false_positives]
            # dicFP = get_frequent_words(X_FP)
            # visualize_frequent_words(dicFP, 30,title='False Positive')
            # #
            # X_FN = X.iloc[false_negatives]
            # print(X_FN.shape)
            # print(X_FN[:2])
            # dicFN = get_frequent_words(X_FN)
            # visualize_frequent_words(dicFN, 30, title='False Negative')

    if settings.SHOW_AVG_SCORES:
        print("none")
        print("precision", precision_N / support_N)
        print("recall", recall_N / support_N)
        print("F1", f1_N / support_N)
        print("support", support_N)
        print("hate")

        print("precision",precision_h/support_h)
        print("recall",recall_h/support_h)
        print("F1",f1_h/support_h)
        print("support",support_h)
        print(cm)
        sns.heatmap(cm,fmt="4",annot=True,xticklabels = ['none','hate'],cmap='Blues', yticklabels = ['none','hate'],cbar=False,square=True)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.savefig("../plots/cm_balanced.pdf", bbox_inches='tight')
        plt.show()

    if settings.EVALUATE_FALSE:
        # analyze wrong prediction
        X_FP = X.iloc[false_positives]
        dicFP = get_frequent_words(X_FP)
        visualize_frequent_words(dicFP, 30, title='False Positive',save_fig=True)
        #

        X_FN = X.iloc[false_negatives]
        print(X_FN.shape)
        print(X_FN[:2])
        dicFN = get_frequent_words(X_FN)
        visualize_frequent_words(dicFN, 30, title='False Negative',save_fig=True)

    if settings.ENTROPY:
        entropyequal = entropyequal.items()
        x, y = zip(*entropyequal)
        x = list(map(lambda x: x/10,list(x)))
        sns.barplot(x=x, y=list(y))
        #plt.title("entropy equals")
        plt.savefig('../plots/confidenceEquals.pdf')
        plt.show()

        entropy_notequal = entropy_notequal.items()
        x, y = zip(*entropy_notequal)
        x = list(map(lambda x: x / 10, list(x)))
        sns.barplot(x=x, y=list(y))
        plt.savefig('../plots/confidenceNotEquals.pdf')
        plt.show()