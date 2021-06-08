from pandas.core.algorithms import unique
import numpy as np
import tensorflow as tf
from tfdeterminism import patch
patch()
SEED = 1234
import random
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)
import os
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

############ my classes ##################
import Preprocessor
########## other packages ################
from collections import Counter
import pandas as pd
import sklearn
from sklearn.metrics import *
import gensim
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Embedding
from keras.layers import LSTM, Flatten
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.utils.vis_utils import plot_model


# ~~~~~setting up the parameters of the model~~~~~~~~~~~~~~~~~~~~
# data settings
loading_option = "relatedness"  # relatedness or info_source or info_type
balancing = True
balancingStrategy = "mix" # "over" for oversampling the minority class, "under" undersampling the majority class, "mix" for using SMOTETomek, a combination of under/over balancing
# network settings
embedding_dim = 300
epochs = 400
batch_size = 256
optimizer = keras.optimizers.Adam()
loss = 'categorical_crossentropy'
early_stop_call = True
load_model = False  # load trained model or train it from start
# ****************************************************
# Auto-config
if loading_option == "relatedness":
    n_class = 2
else:
    n_class = 8

# def graph(history, fig_name="losses.png"):
#     loss_train = history['loss']
#     loss_val = history['val_loss']
#     epoc = range(1, len(loss_train) + 1)
#     plt.plot(epoc, loss_train, 'g', label='Training loss')
#     plt.plot(epoc, loss_val, 'b', label='validation loss')
#     plt.title('Training and Validation loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     fig = plt.gcf()
#     fig.savefig(fig_name)


def model_init(input_shape, vocab_size, n_class, optimizer, loss, embedding_dim, embedding_matrix):
    model_input = Input(shape=(input_shape,), name="text_input")
    emb = Embedding(vocab_size + 1, embedding_dim, weights=[embedding_matrix],
                               input_length=input_shape,
                               trainable=False)(model_input)
    emb = LSTM(128, return_sequences=True)(emb)
    cnn1 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(emb)
    cnn1 = MaxPooling1D(pool_size=int(input_shape/10))(cnn1)
    cnn1 = Flatten()(cnn1)
    model = Dropout(0.5)(cnn1)
    model = Dense(n_class, activation='softmax')(model)
    model = Model(inputs=model_input, outputs=model)
    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


def training(x_train, y_train, epochs, batch_size, early_stop_call=True):
    # callbacks setup
    early_stop = None
    if early_stop_call:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs, validation_split=0.1, verbose=1, shuffle=True,
                            callbacks=[early_stop])
    else:
        history = model.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs, validation_split=0.1, verbose=1, shuffle=True)
    model.save('saved_model')

    return history


def evaluation(x_test, y_test, history, index_before_shuffle=None):
    pr = model.predict(x_test)
    pred = np.argmax(pr, axis=1)
    ytrue = np.argmax(y_test, axis=1)
    report = classification_report(ytrue, pred, output_dict=True)
    report["accuracy"] = {"accuracy": report["accuracy"]}  # cause only accuracy is not a dict and it crashes in df conversion later
    report["loss"] = {"training": min(history["loss"]),"validation": min(history["val_loss"])}
    if loading_option == "relatedness":
        report["auc_macro"] = {"auc_macro": roc_auc_score(ytrue, pr[:, 1], average='macro')}
        report["auc_weighted"] = {"auc_weighted": roc_auc_score(ytrue, pr[:, 1], average='weighted')}
    else:
        report["auc_macro"] = {"auc_macro": roc_auc_score(ytrue, pr, average='macro', multi_class='ovo')}
        report["auc_weighted"] = {"auc_weighted": roc_auc_score(ytrue, pr, average='weighted', multi_class='ovo')}

    # saving results
    predictions = dict()
    for i, value in enumerate(pred):
        index = i  #index_before_shuffle[i]
        predictions[index] = {"prediction": value,
                               "label": ytrue[i]}
    df = pd.DataFrame.from_dict(predictions, orient="index")
    df.to_csv("predictions.csv")
    df = pd.DataFrame.from_dict(report, orient="index")
    df.to_csv("report.csv")
    print(report)


if __name__ == '__main__':

    # loading and preprocessing data
    p = Preprocessor.Pipeline(load_setting=loading_option, n_classes=n_class)
    tweets, labels = p.pipe()
    #labels[:,1] will return the labels in one column for relatedness mode
    #(unique, counts) = np.unique(labels[:,1], return_counts=True) #get the count of each class
    # print(unique) #0,1
    #print(counts) #3352,24581 , highly imbalanced
    xtrain, ytrain, xtest, ytest = p.splitting(tweets, labels)

    if balancing:
        print('x before')
        print(xtrain.shape)
        print(xtrain[0])
        # print('train before')
        # (unique, counts) = np.unique(ytrain[:,1], return_counts=True) 
        # print(unique)
        # print(counts)
        # print('test before',ytest.shape)
        # (unique, counts) = np.unique(ytest[:,1], return_counts=True) 
        # print(unique)
        # print(counts) 
        print('balancing...')
        xtrain, ytrain = p.balancing(xtrain, ytrain,balancingStrategy)
        xtest, ytest = p.balancing(xtest, ytest,balancingStrategy)
        print('x after')
        print(xtrain.shape)
        print(xtrain[0])
        # print('train after')
        # (unique, counts) = np.unique(ytrain[:,1], return_counts=True) 
        # print(unique)
        # print(counts)
        # print('test after',ytest.shape)
        # (unique, counts) = np.unique(ytest[:,1], return_counts=True) 
        # print(unique)
        # print(counts) 
        input('press key to continue')
    tokenizer, xtrain, xtest = p.tokens(train_data=xtrain, test_data=xtest)

    # load Google's pre-trained w2v model
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin', binary=True)

    # build embedding matrix
    count = 0
    embedding_matrix = np.random.random((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        try:
            vec = w2v_model.wv[word]
            count += 1
            embedding_matrix[i] = vec
        except KeyError:  # token is not in the corpus.
            continue
    print("Converted %d words (%d misses)" % (count, len(tokenizer.word_index)-count))

    # training / loading and evaluation
    if load_model:
        model = keras.models.load_model('saved_model')
    else:
        model = model_init(xtrain.shape[1], len(tokenizer.word_index), n_class, optimizer, loss, embedding_dim, embedding_matrix)
        history = training(xtrain, ytrain, epochs, batch_size, early_stop_call=early_stop_call)
        # graph(history.history)
        evaluation(xtest, ytest, history.history)
