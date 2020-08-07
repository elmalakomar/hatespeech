
class Settings:
    def __init__(self):
        # dataset read and manipulation
        self.CSV_PATH = "../data/dataset.csv"
        self.IGNORE_COLUMNS = ['TweetID']
        self.X_LABEL = 'text'
        self.Y_LABEL = 'annotation'

        #function
        self.CLEAN_TEXT = True

        # preprocessing
        self.REMOVE_LINKS = True
        self.REMOVE_NUMBERS = True
        self.EMOJI_TO_TEXT = True
        self.REMOVE_USER_TAGS = True
        self.REMOVE_PUNCTUATION = True
        self.REMOVE_STOPWORDS = True

        self.LOWER_TEXT = True

        self.AUTOCORRECT = False


