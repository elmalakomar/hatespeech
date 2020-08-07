import re
from nltk import TweetTokenizer
from nltk.corpus import stopwords
import string
import emoji
from api.setting import Settings
####


# TODO: verificare se stopwords come 'not' incidano o meno
def remove_stopwords(tokenized):
    stop_words = stopwords.words('english')
    stop_words.extend(['mkr','im','one','kat'])
    stop_words.remove('not')
    tokenized_no_sw = [token for token in tokenized if token not in stop_words]
    return tokenized_no_sw

def remove_punctuation_from_text(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def emo2text(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def tokenize_tweet(text):
    tokenizer = TweetTokenizer()
    return tokenizer.tokenize(text)

def is_usertag(token):
    if token[0] is '@':
        return True
    return False

def clean_tweet(settings, tweet):
    if settings.LOWER_TEXT:
        tweet = tweet.lower()
        #print("lower-> ",tweet)
    if settings.REMOVE_LINKS: #remove links
        tweet = re.sub(r'http\S+', '', tweet)
        #print("link-> ",tweet)
    if settings.REMOVE_NUMBERS:
        tweet = re.sub(r"\d+", "", tweet)
        #print("numbers-> ", tweet)
    if settings.EMOJI_TO_TEXT:
        tweet = emo2text(tweet)
        #print("emoji-> ", tweet)
    tokenized_tweet = tokenize_tweet(tweet)
    tokenized_tweet = [t for t in tokenized_tweet if len(t) > 2]
    #print("tokenized-> ", tokenized_tweet)
    if settings.REMOVE_USER_TAGS:
        #print(list(map(is_usertag, tokenized_tweet)))
        tokenized_tweet = [token for token in tokenized_tweet if not is_usertag(token)]
        #print("user_tags-> ", tokenized_tweet)
    if settings.REMOVE_PUNCTUATION:
        map_result = list(map(remove_punctuation_from_text,tokenized_tweet))
        tokenized_tweet = list(filter(None,map_result))
        #print("punctuation-> ", tokenized_tweet)
    if settings.REMOVE_STOPWORDS:
        tokenized_tweet = remove_stopwords(tokenized_tweet)
        #print("stop_words-> ", tokenized_tweet)
    tweet = ' '.join(token for token in tokenized_tweet).strip()
    return tweet


if __name__ == '__main__':
    settings = Settings()
    tweet = 'RT2: mkr im RT like kat @pythonTAG 3 #pythonHASH Python is a language not 👍.4'
    x = clean_tweet(settings,tweet)
    print(x)
