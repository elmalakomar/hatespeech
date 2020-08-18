
class Settings:
    def __init__(self):
        # settings
        self.RANDOM_SEED = 16

        # dataset read and manipulation
        self.CSV_PATH = "../data/dataset.csv"
        self.IGNORE_COLUMNS = ['TweetID']
        self.X_LABEL = 'text'
        self.Y_LABEL = 'annotation'


        #function
        self.CLEAN_TEXT = True
        self.EVALUATE_FALSE = False
        self.ENTROPY = True

        # preprocessing
        self.LOWER_TEXT = True
        self.CHECK_CONTRACTION = True
        self.REMOVE_LINKS = True
        self.REMOVE_NUMBERS = True
        self.EMOJI_TO_TEXT = True
        self.REMOVE_USER_TAGS = False
        self.REMOVE_PUNCTUATION = True
        self.REMOVE_STOPWORDS = True
        self.STEMMING = True

        self.AUTOCORRECT = False

        ## balancing dataset, undersampling & undersampling
        self.BALANCE_DATA= False
        self.UNDERSAMPLING = False
        self.OVERSAMPLING = False

        #visualization
        self.PRINT_PIE = False
        self.WORDCLOUD = False
        self.MOST_FREQUENT_WORDS = False
        self.SHOW_AVG_SCORES= False

        self.ADD_DATA = False


