
class Settings:
    def __init__(self):
        # dataset read and manipulation
        self.csv_path = "../data/dataset.csv"

        self.ignore_columns = ['TweetID']

        # preprocessing
        self.LOWER_TEXT = True
        self.EMOJI_TO_TEXT = True
        self.REMOVE_NUMBERS = True
        self.REMOVE_PUNCTUATION = True
        self.REMOVE_LINKS = True
        self.AUTOCORRECT = False
        self.REMOVE_HASHTAGS= True

