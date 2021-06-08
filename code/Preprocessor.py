import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from collections import Counter
import json
import sklearn
import imblearn
from sklearn.preprocessing import LabelEncoder


class Pipeline:

    def __init__(self, load_setting, n_classes):
        self.load_setting = load_setting
        self.n_classes = n_classes
        pass

    # label names getter
    def get_label_names(self):
        try: return self.enc.inverse_transform(list(range(self.n_classes)))
        except AttributeError:
            print("There is no attribute enc in Preprocessor. encoder is not used in binary classification")

    # loading and preprocessing data.
    def pipe(self):
        data = self.read_data()
        tweets, labels = self.transform_data(data)
        tweets = self.preprocessing(tweets)
        return tweets, labels

    def preprocessing(self, tw):
        tw = [nltk.re.sub(r"http\S+", "link", text) for text in tw]  # replacing links: <LINK>
        tw = [nltk.re.sub(r"@\S+", "tag", text) for text in tw]  # replacing tags: <TAG>
        tw = [nltk.re.sub(r'[0-9]+', " digits ", text) for text in tw]  # replacing tags: digit
        tw = [nltk.re.sub(r"[\'|\"]", " ", text) for text in tw]  # removing ' and "
        tw = [nltk.re.sub(r"\b\w\b", "", text) for text in tw]  # remove single character words
        text_to_list = tf.keras.preprocessing.text.text_to_word_sequence
        tw = [text_to_list(sentence) for sentence in tw]
        stopwords = set(nltk.corpus.stopwords.words('english'))
        tw = [[word for word in words if word not in stopwords] for words in tw]
        tw = [" ".join(tweet) for tweet in tw]
        return np.asarray(tw)

    # if called only for test: train_data is None --> so it returns None xtrain var.
    def tokens(self, test_data, train_data=None, tokenizer=None, maxim=None):
        xtrain = None
        self.maxim = maxim
        if train_data is not None:
            tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
            tokenizer.fit_on_texts(train_data)
            train_indexes = tokenizer.texts_to_sequences(train_data)
            if self.maxim is None:
                self.maxim = max([len(i) for i in train_indexes])
            xtrain = tf.keras.preprocessing.sequence.pad_sequences(train_indexes, maxlen=self.maxim)
        test_indexes = tokenizer.texts_to_sequences(test_data)
        xtest = tf.keras.preprocessing.sequence.pad_sequences(test_indexes, maxlen=self.maxim)
        return tokenizer, xtrain, xtest

    # def balance(self,tweets,labels):
    #     labels = np.argmax(labels, axis=1) #oneHot to 1 column
    #     tweets = tweets.reshape(-1,1) #because x is single feature
    #     ros = imblearn.over_sampling.RandomOverSampler(random_state=0) #random state is set to a value for testing consistency
    #     x_res, y_res = ros.fit_resample(tweets,rawLabels) #resampling
    #     y_res = y_res.astype(int)
    #     x_res = x_res.flatten()
    #     hotLabels = np.zeros((y_res.size,y_res.max()+1))
    #     hotLabels[np.arange(y_res.size),y_res] = 1
    #     return x_res,hotLabels

        

    def balancing(self, tweets, labels, strategy="under"):
        enc = sklearn.preprocessing.OneHotEncoder()
        vec = sklearn.feature_extraction.text.CountVectorizer
        tweets = tweets.reshape((-1,1))
        tweets_enc = enc.fit_transform(tweets)
        tweets_enc = tweets_enc.toarray()
        labels = np.argmax(labels, axis=1)
        if strategy == "under":
            undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority')
            y_res, x_res = undersample.fit_resample(tweets_enc, labels)
        elif strategy == "over":
            oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority')
            y_res, x_res = oversample.fit_resample(tweets_enc, labels)
        else:
            mix = imblearn.combine.SMOTEENN(random_state=0)
            y_res, x_res = mix.fit_resample(tweets_enc, labels)

        x_res = x_res.flatten()
        y_res = enc.fit_transform(y_res)
        return x_res, y_res
    # def balancing(self, tweets, labels):
    #     new_i = np.random.permutation(len(labels))
    #     labels = labels[new_i]
    #     tweets = tweets[new_i]

    #     yy = []
    #     xx = []
    #     y = np.argmax(labels, axis=1)
    #     for i, t in enumerate(y):
    #         if t == 0:
    #             yy.append(t)
    #             xx.append(tweets[i])
    #     counter = 0
    #     for i, t in enumerate(y):
    #         if t == 1:
    #             yy.append(t)
    #             xx.append(tweets[i])
    #             counter += 1
    #         if counter >= len(y) - y.sum():
    #             break
    #     yy = np.asarray(yy)
    #     xx = np.asarray(xx)
    #     new_i = np.random.permutation(len(yy))
    #     labels = yy[new_i]
    #     tweets = xx[new_i]
    #     return tweets, tf.keras.utils.to_categorical(labels, num_classes=len(Counter(labels)), dtype='float32')

    def splitting(self, tweets, labels):
        split = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1,
                                                                   test_size=0.2,
                                                                   random_state=20)
        train_idx, test_idx = list(split.split(tweets, labels))[0]
        xtrain = tweets[train_idx]
        ytrain = labels[train_idx]
        xtest = tweets[test_idx]
        ytest = labels[test_idx]
        return xtrain, ytrain, xtest, ytest

    def transform_data(self, data):
        tweets = data.iloc[:, 1].to_numpy()  # tweets to nd array
        if self.load_setting == "relatedness":
            labels = data.iloc[:, 4].to_numpy()  # informativeness is the label column.
        elif self.load_setting == "info_source":
            labels = data.iloc[:,2]
        elif self.load_setting == "info_type":
            labels = data.iloc[:,3]

        if self.load_setting != "relatedness":
            self.enc = LabelEncoder()
            labels = self.enc.fit_transform(labels)
        else:
            # convert label values: not related --> 0 , related --> 1
            for i, label in enumerate(labels):
                if label == "Not applicable":
                    labels[i] = 0
                elif label == "Not related":
                    labels[i] = 0
                else:
                    labels[i] = 1
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(Counter(labels)), dtype='float32')
        return tweets, labels

    def read_data(self):
        try:
            data = pd.read_csv("data/data.csv")
        except FileNotFoundError:
            print("there is no dataset in ")
            list_subfolders = sorted([f.name for f in os.scandir("data") if
                                      f.is_dir()])  # scans the folder "data" to get a list of all subfolders
            # data is the dataframe for all concatenated datasets , initialized with the first crisis data
            data = pd.read_csv("data/" + list_subfolders[0] + "/" + list_subfolders[0] + "-tweets_labeled.csv")
            for i, crisis in enumerate(list_subfolders):
                if i == 0: continue
                crisis_data = pd.read_csv(
                    "data/" + list_subfolders[i] + "/" + list_subfolders[i] + "-tweets_labeled.csv")
                data = pd.concat([data, crisis_data], sort=False, ignore_index=True)
        return data


if __name__ == '__main__':
    p = Pipeline()
    tweets, labels = p.pipe()
    print()
