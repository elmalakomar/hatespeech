import re
from nltk import TweetTokenizer
from nltk.corpus import stopwords
import string
import emoji
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

from api.setting import Settings
####

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot",
                   "can't've": "cannot have", "'cause": "because", "could've": "could have",
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not",
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did",
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                   "I'll've": "I will have","I'm": "I am", "I've": "I have",
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                   "i'll've": "i will have","i'm": "i am", "i've": "i have",
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                   "it'll": "it will", "it'll've": "it will have","it's": "it is",
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have",
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                   "she's": "she is", "should've": "should have", "shouldn't": "should not",
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is",
                   "there'd": "there would", "there'd've": "there would have","there's": "there is",
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                   "they'll've": "they will have", "they're": "they are", "they've": "they have",
                   "to've": "to have", "wasn't": "was not", "we'd": "we would",
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                   "we're": "we are", "we've": "we have", "weren't": "were not",
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                   "what's": "what is", "what've": "what have", "when's": "when is",
                   "when've": "when have", "where'd": "where did", "where's": "where is",
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                   "who's": "who is", "who've": "who have", "why's": "why is",
                   "why've": "why have", "will've": "will have", "won't": "will not",
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                   "you'll've": "you will have", "you're": "you are", "you've": "you have",}

# TODO: verificare se stopwords come 'not' incidano o meno
def remove_stopwords(tokenized):
    stop_words = stopwords.words('english')
    stop_words.extend(['really','twitter','mykitchenrules','mkr','kat','like',
                       'would','get','think',
                       'one','people','want',
                       'know','see','mkr2015',
                       'andre','even','got',
                       'also','annie', 'fuck'])
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
    tweet = str(tweet)

    if settings.LOWER_TEXT:
        tweet = tweet.lower()
        tweet = re.sub(r'(.)\1+', r'\1\1',tweet)
        #print("lower-> ",tweet)
    if settings.CHECK_CONTRACTION:
        tweet = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in tweet.split(" ")])
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
    if settings.STEMMING:
        stemmer = SnowballStemmer("english")
        tokenized_tweet = [stemmer.stem(token) for token in tokenized_tweet]
    tweet = ' '.join(token for token in tokenized_tweet).strip()
    #tweet = re.sub(r'(.)\1+', r'\1\1',tweet)
    return tweet
