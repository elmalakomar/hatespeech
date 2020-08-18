import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns


def get_balanced_data(df,y_label):
    print("balancing data")
    none_data = pd.DataFrame([row[1] for row in df.iterrows() if row[1][y_label] == 'none'])
    hate_data = pd.DataFrame([row[1] for row in df.iterrows() if row[1][y_label] != 'none'])
    none_data = none_data.sample(frac=1,random_state=16).head(int(hate_data.shape[0]*1.5))
    return none_data.append(hate_data,ignore_index=True)

def csv2data(path, x_label, y_label,
             ignore_columns=[],
             balance_data = False,
             random = 16):

    data = pd.read_csv(path)

    data.drop(ignore_columns, axis=1, inplace=True)

    if balance_data:
        data = get_balanced_data(data,y_label)
    X = data[x_label]
    Y = data[y_label]
    return data,X,Y

def get_frequent_words(data):
    d = defaultdict(int)
    for tweet in data:
        tweet = str(tweet)
        for word in tweet.split():
            d[word] += 1
    return d.items()

def visualize_frequent_words(dic,top,title = 'Most frequent words',save_fig = False,path = '../plots/'):
    top_list = sorted(dic, key=lambda x: x[1], reverse=True)[:top]
    print(top_list)
    x, y = zip(*top_list)
    sns.barplot(x=list(y),y=list(x))
    plt.title(title)
    title = title.lower()
    if save_fig:
        plt.savefig(path+title.replace(" ", "")+'.pdf')
    plt.show()

def worldCloud(X,Y,x_label,y_label, label=1):
    data = X.to_frame(x_label).join(Y)
    words = ''
    for row in data[data[y_label] == label][x_label]:
        words += row + ' '
    wordcloud = WordCloud(max_words=100, min_font_size=10,background_color="white", width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('../plots/'+str(label)+ '.pdf',bbox_inches='tight')
    plt.show()

