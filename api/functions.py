import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns


def get_balanced_data(df,y_label):
    print("balancing data")
    none_data = pd.DataFrame([row[1] for row in df.iterrows() if row[1][y_label] == 'none'])
    hate_data = pd.DataFrame([row[1] for row in df.iterrows() if row[1][y_label] != 'none'])
    none_data = none_data.sample(frac=1,random_state=16).head(hate_data.shape[0])
    return none_data.append(hate_data,ignore_index=True)

def csv2data(path,x_label, y_label, ignore_columns=[], balance_data = False):
    data = pd.read_csv(path)
    data.drop(ignore_columns, axis=1, inplace=True)
    if balance_data:
        data = get_balanced_data(data,y_label)
    data.sample(frac=1,random_state=16)
    X = data[x_label]
    Y = data[y_label]
    print(data.shape)
    return data,X,Y

def get_frequent_words(data):
    d = defaultdict(int)
    for tweet in data:
        for word in tweet.split():
            d[word] += 1
    return d.items()

def visualize_frequent_words(dic,top,title = 'foo'):
    top_list = sorted(dic, key=lambda x: x[1], reverse=True)[:top]
    print(top_list)
    x, y = zip(*top_list)
    sns.barplot(x=list(y),y=list(x))
    plt.title(title)
    plt.savefig('../plots/'+title.replace(" ", "")+'.pdf')
    plt.show()

def worldCloud(X,Y,label=1):
    data = X.to_frame('tweet').join(Y)
    print(type(data), data.shape)
    print(data.head())
    words = ''
    for row in data[data['annotation'] == label]['tweet']:
        words += row + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

