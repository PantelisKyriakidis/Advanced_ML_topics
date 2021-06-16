import numpy as np
SEED = 1234
import random
random.seed(SEED)
np.random.seed(SEED)
import os
os.environ["PYTHONHASHSEED"] = str(SEED)

############ my classes ##################
import Preprocessor
########## other packages ################
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import *
import matplotlib.pyplot as plt
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from modAL.models import Committee
from modAL.disagreement import vote_entropy_sampling, max_disagreement_sampling


# ~~~~~setting up the parameters of the model~~~~~~~~~~~~~~~~~~~~
# data settings
loading_option = "relatedness"
balancing = True
max_features = 1000
# training settings
clr = 'nb'  # classifier 'nb', 'svm'
committee_flag = False  # query by committee?

# Auto-config
if loading_option == "relatedness":
    n_class = 2
else:
    n_class = 8

scores = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
f1_general = {'us': [], 'es': [], 'ms': [], 'rs': []}


# custom sampling strategy
def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return query_idx, X_pool[query_idx]


def create_learners(qs, X_initial, y_initial):
    if clr == 'nb':
        learner1 = ActiveLearner(
            estimator=MultinomialNB(alpha=1),
            query_strategy=qs,
            X_training=X_initial, y_training=y_initial)
        learner2 = ActiveLearner(
            estimator=MultinomialNB(alpha=3),
            query_strategy=qs,
            X_training=X_initial, y_training=y_initial)
        learner3 = ActiveLearner(
            estimator=MultinomialNB(alpha=2),
            query_strategy=qs,
            X_training=X_initial, y_training=y_initial)
    else:
        learner1 = ActiveLearner(
            estimator=SVC(gamma='scale', kernel='linear', probability=True),
            query_strategy=qs,
            X_training=X_initial, y_training=y_initial)
        learner2 = ActiveLearner(
            estimator=SVC(gamma='scale', kernel='rbf', probability=True),
            query_strategy=qs,
            X_training=X_initial, y_training=y_initial)
        learner3 = ActiveLearner(
            estimator=SVC(gamma='scale', kernel='poly', probability=True),
            query_strategy=qs,
            X_training=X_initial, y_training=y_initial)

    return learner1, learner2, learner3


# first sampling before active learning.
def init_data_sampling(X_train, y_train):
    # assemble initial data
    n_initial = 10  # train on only 10 random samples
    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
    X_initial = X_train[initial_idx]
    y_initial = y_train[initial_idx]

    # generate the pool
    # remove the initial data from the training dataset
    X_pool = np.delete(X_train, initial_idx, axis=0)[:6000]
    y_pool = np.delete(y_train, initial_idx, axis=0)[:6000]
    return X_initial, y_initial, X_pool, y_pool


def update_results(y_test,y_predicted,i):
    scores['acc'].append(accuracy_score(y_test, y_predicted))
    scores['rec'].append(recall_score(y_test, y_predicted, average='macro'))
    scores['prec'].append(precision_score(y_test, y_predicted, average='macro'))
    scores['f1'].append(f1_score(y_test, y_predicted, average='macro'))
    f1_general[i].append(f1_score(y_test, y_predicted, average='macro'))


# resets scores after one loop
def reset_tables():
    scores['acc'].clear()
    scores['prec'].clear()
    scores['rec'].clear()
    scores['f1'].clear()


def plots(i):
    fig, ax = plt.subplots()
    ax.plot(scores['rec'])
    ax.plot(scores['prec'])
    ax.plot(scores['f1'])
    ax.legend(['Recall', 'Precision', 'F1-score'])
    ax.set_ylim(bottom=0, top=1)

    if i == 'us': title = "Uncertainty sampling"
    elif i == 'ms': title = "Margin sampling"
    elif i == 'es': title = "Entropy sampling"
    elif i == 'rs': title = "Random sampling"
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Performance')
    fig.savefig(title + ".png")
    plt.show()


def f1_plot():
    fig, ax = plt.subplots()
    ax.plot(f1_general['us'])
    ax.plot(f1_general['ms'])
    ax.plot(f1_general['es'])
    if committee_flag: ax.plot(f1_general['rs'])
    legend_list =['uncertainty sampling', 'margin sampling', 'entropy sampling']
    if committee_flag: legend_list.append('random sampling')
    ax.legend(legend_list)
    ax.set_ylim(bottom=0, top=1)
    ax.set_title('Overall F1-score')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Performance')
    fig.savefig("Overall.png")
    plt.show()


if __name__ == '__main__':

    # data pipeline
    p = Preprocessor.Pipeline(load_setting=loading_option, n_classes=n_class)
    tweets, labels = p.pipe()
    xtrain, ytrain, xtest, ytest = p.splitting(tweets, labels)
    if balancing:
        xtrain, ytrain = p.balancing(xtrain, ytrain, 'under')
        xtest, ytest = p.balancing(xtest, ytest, 'under')
    vectorizer = CountVectorizer(max_features=max_features)
    xtrain = vectorizer.fit_transform(xtrain).toarray()
    xtest = vectorizer.transform(xtest).toarray()
    ytrain = np.argmax(ytrain,axis=1)
    ytest = np.argmax(ytest, axis=1)

    # loop over different query strategies
    q_strategies = ['us', 'es', 'ms']
    if committee_flag: q_strategies.append("rs")  # random sampling did not work without committee
    for i in q_strategies:
        if i == 'us': qs = uncertainty_sampling
        elif i == 'es': qs = entropy_sampling
        elif i == 'ms': qs = margin_sampling
        elif i == 'rs': qs = random_sampling

        X_initial, y_initial, X_pool, y_pool = init_data_sampling(xtrain, ytrain)

        # creating learners. if committee flag == True then create a committee with a subset with the returned learners.
        # else: use only the first learner
        learner1, learner2, learner3 = create_learners(qs, X_initial, y_initial)
        if committee_flag:
            learner = Committee(
                learner_list=[learner1, learner3],
                query_strategy=vote_entropy_sampling
            )
        else:
            learner = learner1

        # the active learning loop
        update_results(ytest, learner.predict(xtest), i)
        n_queries = 10
        for idx in range(n_queries):
            query_idx, query_instance = learner.query(X_pool, n_instances=300)
            learner.teach(
                X=X_pool[query_idx], y=y_pool[query_idx]
            )
            # remove queried instance from pool
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx, axis=0)
            print(idx)
            update_results(ytest, learner.predict(xtest), i)
        plots(i)
        reset_tables()
    f1_plot()
