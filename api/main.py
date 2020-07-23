import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from wordcloud import WordCloud
import nltk
from nltk.stem import WordNetLemmatizer


from api.functions import most_frequent_words
from api.setting import Settings
from api.classification import run

def data_cleaning(tweet):
    tweet = tweet.lower()
    tokens = nltk.tokenize.word_tokenize(tweet)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    separator = ' '
    result = separator.join(tokens)
    return result

def visualizeWordCloud(label):
    #get all words from spam msgs
    words = ''
    for msg in data[data['annotation'] == label]['text']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    return words

def visualizeWordCloudFromX():
    #get all words from spam msgs
    words = ''
    for msg in X:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    return words


if __name__ == '__main__':
    settings = Settings()
    run(settings)

    # data = pd.read_csv("../data/dataset.csv")
    # data.drop(['TweetID'],axis = 1, inplace=True)
    # most_frequent_words(data,'sexism')
    # most_frequent_words(data, 'none')
    # most_frequent_words(data, 'racism')

# if __name__ == '__main__':
#     wordnet_lemmatizer = WordNetLemmatizer()
#     data = pd.read_csv("../data/dataset.csv")
#     data.drop(['TweetID'],axis = 1, inplace=True)
#     print(data.head(10))
#
#     stopwords = set(w.rstrip() for w in open("../data/stopwords.txt"))
#     stopwords.discard('')
#     print(stopwords)
#
#     X = data['text']
#     Y = data['annotation']
#     Y = Y.map({'none': 0, 'racism': 1, 'sexism': 1})
#     print(X[0])
#     X = X.apply(lambda x: data_cleaning(x))
#     print(X[0])
#
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(X)
#     print("Input X after preprocessing: {}".format(X.shape))
#
#     Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, shuffle=True, random_state=42)
#
#     # apply ML on the data
#     model = MultinomialNB()
#     model.fit(Xtrain, Ytrain)
#     pred = model.predict(Xtest)
#
#     # evaluate
#     print("Classification rate for NB:", model.score(Xtest, Ytest))
#
#     print("confusion matrix:")
#     print(metrics.confusion_matrix(Ytest, pred))
#
#     print("classification report:")
#     print(metrics.classification_report(Ytest, pred, target_names=["none", "hate"]))
#
#     #text pre-processing
#     words = visualizeWordCloud('none')

