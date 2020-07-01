import pandas as pd


# get dataframe (ID,ann1,ann2,ann3,ann4)
#   return dataframe(ID,ann)
def get_majority_annotation(dataframe):
    result = []
    for _, row in dataframe.iterrows():
        result.append(row.iloc[1:].value_counts().idxmax())
    return result


def get_maj_dataset():
    css = pd.read_csv("../data/NLP+CSS_2016.csv", delimiter=";")
    print(css.head())
    majority_list = get_majority_annotation(css)
    print(len(majority_list))
    css = css.assign(annotation=majority_list)
    css.drop(columns=["Expert", "Amateur_0", "Amateur_1", "Amateur_2"], inplace=True)
    print(css.head())
    css.to_csv("../data/new_css.csv")


def merge_datasets():
    css = pd.read_csv("../data/new_css.csv")
    srw = pd.read_csv("../data/NAACL_SRW_2016.csv", names=["TweetID", "annotation"])
    df = css[['TweetID', 'annotation']].append(srw)
    df = df.drop_duplicates('TweetID')
    df.to_csv("../data/hatespeech.csv", index=False, header=["TweetID", "annotation"])

def join_datasets():
    annotations = pd.read_csv("../data/hatespeech.csv")
    print(annotations.shape)
    text = pd.read_csv("../data/hate_dataset.csv")
    print(text.shape)
    print(len(set(text["TweetID"]).difference(set(annotations["TweetID"]))))
    joined = text.join(annotations.set_index('TweetID'), on='TweetID')
    print(joined.head())
    print(joined.isna().sum())
    joined.to_csv("../data/dataset.csv", index=False, header=["TweetID", "text", "annotation"])

if __name__ == '__main__':
    df = pd.read_csv("../data/dataset.csv")
    print(df.head())


