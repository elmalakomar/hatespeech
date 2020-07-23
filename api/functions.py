import re
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
import emot
import demoji
import emoji
from nltk.tokenize import TweetTokenizer
from wordsegment import load, segment
load()

def most_frequent_words_label(data, label):
    d = {}
    for tweet in data[data['annotation'] == label]['text']:
        for word in tweet.split():
            if word not in d:
                d[word] = 0
            d[word] += 1
    sort_orders = sorted(d.items(), key=lambda x: x[1], reverse=True)
    print(sort_orders[0:15])

def most_frequent_words(data):
    d = {}
    for tweet in data:
        for word in tweet.split():
            if word not in d:
                d[word] = 0
            d[word] += 1
    d = sorted(d.items(), key=lambda x: x[1], reverse=True)
    print(d)
    return d

# -- PRE PROCESSING -- #

def convert_emojis(text,emojis):
    to_append = ' '.join(emojis['mean'])
    to_append = to_append.replace(":", " ").replace("_", " ")
    text = text + " " + to_append
    for emoji in emojis['value']:
        #print(emoji)
        text = text.replace(emoji, ' ')
    return text

def emo2text(text):
    text = emoji.demojize(text, delimiters=(" ", " "))
    #
    # emojis = emot.emoji(text)
    # if emojis['flag'] is True:
    #     text = convert_emojis(text, emojis)
    return text

def tokenize_tweet(text):
    tokenizer = TweetTokenizer()
    return tokenizer.tokenize(text)

# function that removes tags, RT, links
def clean_tweet(tweet):

    #remove links
    tweet = re.sub(r'http\S+', '', tweet)
    #print("LINKS->",tweet)
    tweet = emo2text(tweet)
    #print("EMOJI->",tweet)
    tweet = tokenize_tweet(tweet)
    #print("TOKENIZED->",tweet)
    # remove user-tags
    tweet = remove_user_tags(tweet)
    #print("USER_TAGS->",tweet)
    #remove punctuation
    tweet = remove_punctuation(tweet)
    #print("PUNCTUATION->",tweet)
    #tweet = [t for t in tweet if t != 'RT'] TODO: rt metterlo nelle stopwords
    tweet = ' '.join(token for token in tweet).strip()
    #print(tweet)
    #tweet = ' '.join(segment(tweet)).strip()
    return tweet

def remove_punctuation_text(text):
    punctuations = ',.!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
    result = ''
    for c in text:
        if c in punctuations:
            result = result+' '
        else:
            result = result+c
    result = result.strip()
    return result

def remove_punctuation(tokenized_tweet):
    result = []
    for token in tokenized_tweet:
        temp = remove_punctuation_text(token)
        result.append(temp)
    result = [x for x in result if x != '' and x != ' ']
    return result

def remove_user_tags(tokenized_tweet):
    result = []
    for token in tokenized_tweet:
        if '@' not in token:
            result.append(token)
    return result


def remove_stopwords():
    pass

if __name__ == '__main__':
    tweet = "RT @freebsDgirl, #camERAshy !w? -__not_v  #mainecoon 📷🎥 https://t.co/Sqz1otVsDC telling #bradpitt http://ww.aa.com jira/bitbucket"
    print(clean_tweet(tweet))
    #print(remove_punctuation(["!w?","#camerashy"]))