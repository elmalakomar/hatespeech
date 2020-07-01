import tweepy
import twitter_credentials
import pandas as pd

def get_auth():
    auth = tweepy.OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
    auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    return api

## prendo lista di tweet e torno gli stati.
def get_statuses(list_tweetID):
    api = get_auth()
    list_status = api.statuses_lookup(list_tweetID, map_=True)
    return list_status


if __name__ == '__main__':
    hate = pd.read_csv("data/hatespeech.csv")
    final_dataset = list()
    tweetsID = hate["TweetID"].tolist()

    for n in range(0,len(tweetsID),100):
        list_status = get_statuses(tweetsID[n:n+100]) #lista degli id
        for tweet in list_status:
            try:
                id = tweet.id
                text = tweet.text
            except AttributeError:
                continue
            final_dataset.append({"TweetID":tweet.id,
                                  "text":tweet.text})

        temp = pd.DataFrame.from_records(final_dataset)
        temp.to_csv("data/hate_dataset.csv",index=False)
