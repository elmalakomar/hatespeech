import pandas as pd
from wordsegment import load
load()
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

def csv2data(path,x_label, y_label, ignore_columns=[]):
    data = pd.read_csv(path)
    data.drop(ignore_columns, axis=1, inplace=True)
    X = data[x_label]
    Y = data[y_label]
    return data,X,Y

def get_frequent_words(data):
    d = defaultdict(int)
    for tweet in data:
        for word in tweet.split():
            d[word] += 1
    return d.items()

def visualize_frequent_words(dic,top):
    top_list = sorted(dic, key=lambda x: x[1], reverse=True)[:top]
    print(top_list)
    x, y = zip(*top_list)
    sns.barplot(x=list(y),y=list(x))
    plt.show()


def visualizeWordCloud(data,label):
    words = ''
    for msg in data[data['annotation'] == label]['text']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    return words

