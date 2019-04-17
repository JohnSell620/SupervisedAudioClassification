import operator
import numpy as np
import pandas as pd
from scipy import signal

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold

from utils import get_sample_data, svd_spectrogram, histogram_separability,\
                    plot_sv_spectrum, plot_mode_histogram, plot_model_scores

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# Classifier names
clf_names = ['SVM', 'NB', 'Decision Tree', 'K Nearest Neighbors', 'Neural Net',
             'Gaussian Process', 'Random Forest', 'AdaBoost', 'LDA', 'QDA']

def train_and_evaluate(X, y):
    """
    Train and evaluate machine learning models on audio signal data. Random
    Forest had best for tests 1 and 3. Naive Bayes had best for test 2.

    :param X: Data examples (right singular vectors, modes).
    :param y: Data labels (classes).
    :return scores: Array containing accuracies of cross-validation for models.
    :rtype: array_type
    """
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.15, random_state=42)

    classifiers = [
            SVC(kernel='poly', degree=4, C=1.0, gamma='auto'),
            GaussianNB(),
            DecisionTreeClassifier(max_depth=3),
            KNeighborsClassifier(),
            MLPClassifier(alpha=1),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            RandomForestClassifier(bootstrap=False),
            AdaBoostClassifier(),
            LinearDiscriminantAnalysis(solver='svd'),
            QuadraticDiscriminantAnalysis()]

    # Evaluate using K-Fold, Stratified K-Fold and Bootstrap.
    kfold = KFold(n_splits=10, random_state=7)
    stratified_kfold = StratifiedKFold(n_splits=10, random_state=3)
    # Deprecated
    # bootstrap = Bootstrap(n=len(X_train), n_bootstraps=5, n_train=0.85,
    #                       random_state=0)
    bs_scores = []
    bootstrap_iter = 10
    for clf in classifiers:
       for j in range(bootstrap_iter):
           X_, y_ = resample(X_train, y_train)
           clf.fit(X_, y_)
           y_pred = clf.predict(X_test)
           accuracy = accuracy_score(y_pred, y_test)
           bs_scores.append(accuracy)
    bs_scores = np.asarray(bs_scores)

    scores = np.zeros(len(clf_names))
    j = 0
    scores_struct = {}
    for name, clf in zip(clf_names, classifiers):
        nk_scores = cross_val_score(clf, X_train, y_train, cv=kfold)
        sk_scores = cross_val_score(clf, X_train, y_train, cv=stratified_kfold)
        # bs_scores = cross_val_score(clf, X_train, y_train, cv=bootstrap)
        bs_scores = []
        for j in range(bootstrap_iter):
            X_, y_ = resample(X_train, y_train)
            clf.fit(X_, y_)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_pred, y_test)
            bs_scores.append(accuracy)
        bs_scores = np.asarray(bs_scores)

        # result = 'Accuracy (Naive CV): %0.2f (+/- %0.2f)\n' % (nk_scores.mean(), nk_scores.std() * 2)
        # result += 'Accuracy (Stratified CV): %0.2f (+/- %0.2f)\n' % (sk_scores.mean(), sk_scores.std() * 2)
        # result += 'Accuracy (Bootstrapped): %0.2f (+/- %0.2f)\n' % (bs_scores.mean(), bs_scores.std() * 2)
        # scores_struct[name] = result + ' {} modes'.format(X.shape[1])
        # print('%0.2f (+/- %0.2f)' % (sk_scores.mean(), sk_scores.std() * 2))

        # Take stratified K-fold.
        scores[j] = sk_scores.mean()
        j += 1

    # Select the top 3 performing classifiers and score on X_test, y_test
    # sorted_scores = sorted(scores_struct.items(), key=lambda x: x[1], reverse=reverse)
    # for k, v in sorted_scores:
    #     score = ''

    return scores


def score_models(test, separability_measure='MIS'):
    """
    Score the ML models on processed data and plot mode histograms.

    :param test: String indicating specific data set of test.
    :param separability_measure: String indicating the separability measure to
        use.
    :return scores: Numpy array of model accuracies.
    :rtype: array_type
    """
    X, y = get_sample_data(test)
    _, s, vh = svd_spectrogram(X)

    # Plot mode histograms to examine separability.
    plot_mode_histogram(vh[:,:6], range(6), test)
    plot_mode_histogram(vh[:,6:12], range(6,12), test)
    plot_mode_histogram(vh[:,12:18], range(12, 18), test)
    plot_mode_histogram(vh[:,18:24], range(18, 24), test)
    plot_sv_spectrum(s/np.max(s), test)

    # Compute separability measures of histograms.
    hsmodes = histogram_separability(vh, separability_measure)

    # Train and evaluation models on a taken number of modes. Increase the
    # number of modes by 2 until all modes are used.
    number_clf = 10
    mode_range = range(2, len(hsmodes)+2, 2)
    scores = np.zeros((number_clf, len(mode_range)))
    for j, m in enumerate(mode_range):
        Vh = np.asarray([vh[:,i] for i in hsmodes[:m]])
        scores_struct = train_and_evaluate(Vh.T, y)
        scores[:,j] = scores_struct

    return scores


def run_test(test, separability_metric):
    """
    Scores models and plots results versus the number of modes used in the
    training data.

    :param test: String indicating which test is run.
    :param separability_metric: String indication which dispersion metric to use.
    :return: Sorted list of accuracy scores for each model.
    :rtype: list(float)
    """
    S = score_models(test, separability_metric)
    plot_model_scores(S, test, clf_names, separability_metric)
    max_scores = {}
    for j in range(S.shape[0]):
        I, M = max(enumerate(S[j]), key=operator.itemgetter(1))
        max_scores[clf_names[j]] = (M, (I+1)*2)
    return sorted(max_scores.items(), key=lambda x: x[1], reverse=True)


def main(test):
    tsad = run_test(test, 'SAD')
    tssd = run_test(test, 'SSD')
    tpcc = run_test(test, 'PCC')
    tmis = run_test(test, 'MIS')
    return [tsad, tssd, tpcc, tmis]


if __name__ == '__main__':
    results_t1 = main('test1')
    results_t2 = main('test2')
    results_t3 = main('test3')
