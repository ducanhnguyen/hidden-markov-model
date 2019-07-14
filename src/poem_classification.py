'''
The training time is too long with >=100 lines of poem.
Also, it gives low accuracy.
Maybe HMM is not suitable for classification when the number of symbols is too large.
'''
import sys
from multiprocessing import Process

import numpy as np
from google.colab import drive
from sklearn.utils import shuffle

from src.hmmd_tf2 import HMMD_TF


# @DeprecationWarning
def read_data(path, label):
    X = []
    y = []
    poem = open(path).read()

    for line in poem.lower().split("\n"):
        X.append(line)
        y.append(label)

    return X, y


def train_classifier(poem1, poem2, weight_hmm1, weight_hmm2):
    '''
    Multithread binary training.

    Each author will have a separate HMM.

    :param poem1: path of the first poem file, or many poems could be put in a single file.
    :param poem2: path of the second poem file
    :param weight_hmm1: the location where the value of hyperparameters of HMM should be stored during training.
    :param weight_hmm2: the location where the value of hyperparameters of HMM should be stored during training.
    :return:
    '''
    # the number of line of poems will be analyzed to train a HMM.
    # should be a small value. The value may be highers, but it takes time.
    limit = 100

    # two model has the same number of hidden units.
    n_hidden_states = 20

    # train model 1
    X1, y1 = read_data(poem1, label=0)
    X1_train = X1[:limit]

    hmm1 = HMMD_TF()
    sequences1, vocabulary1 = hmm1.load_data(X1_train, split_level='WORD')

    process1 = Process(target=hmm1.fit,
                       args=(sequences1, vocabulary1, weight_hmm1, n_hidden_states, ''))
    process1.start()

    # train model 2
    X2, y2 = read_data(poem2, label=1)
    X2_train = X2[:limit]

    hmm2 = HMMD_TF()
    sequences2, vocabulary2 = hmm2.load_data(X2_train, split_level='WORD')
    process2 = Process(target=hmm2.fit,
                       args=(sequences2, vocabulary2, weight_hmm2, n_hidden_states,
                             '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t'))
    process2.start()

    # join all processes
    process1.join()
    process2.join()


def test(poem1, poem2, weight_hmm1, weight_hmm2):
    # get data
    X1, y1 = read_data(poem1, label=0)
    upper_limit = 110
    lower_limit = 100
    X1_test = X1[lower_limit:upper_limit]
    y1_test = y1[lower_limit:upper_limit]

    X2, y2 = read_data(poem2, label=1)
    X2_test = X2[lower_limit:upper_limit]
    y2_test = y2[lower_limit:upper_limit]

    # create the final test
    X_test = np.concatenate([X1_test, X2_test])
    y_test = np.concatenate([y1_test, y2_test])
    X_test, y_test = shuffle(X_test, y_test)

    # compute likelihood from model 1
    hmm1 = HMMD_TF()
    hmm1.read(A_path=weight_hmm1 + "logA.csv", B_path=weight_hmm1 + "logB.csv",
              pi_path=weight_hmm1 + "logpi.csv",
              vocabulary_path=weight_hmm1 + "vocabulary.csv")
    likelihood_arr1 = hmm1.compute_likelihood(X_test, split_level='WORD')
    print(f'log likelihood_arr1: {likelihood_arr1}')

    # compute likelihood from model 2
    hmm2 = HMMD_TF()
    hmm2.read(A_path=weight_hmm2 + "logA.csv", B_path=weight_hmm2 + "logB.csv",
              pi_path=weight_hmm2 + "logpi.csv",
              vocabulary_path=weight_hmm2 + "vocabulary.csv")
    likelihood_arr2 = hmm2.compute_likelihood(X_test, split_level='WORD')
    print(f'log likelihood_arr2: {likelihood_arr2}')

    # compute score
    yhat = np.argmax([likelihood_arr1, likelihood_arr2], axis=0)
    print(f'True ground: {y_test}')
    print(f'Prediction : {yhat}')
    score = np.mean(yhat == y_test)
    print(f'accuracy = {score}')


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)

    GOOGLE_COLAB = False

    if GOOGLE_COLAB:
        # run on google colab, but it is still too long
        drive.mount('/content/drive')

        poem1 = "/content/drive/My Drive/Colab Notebooks/poem/nguyen-binh.txt"
        poem2 = "/content/drive/My Drive/Colab Notebooks/poem/truyen_kieu.txt"

        weight_hmm1 = "/content/drive/My Drive/Colab Notebooks/poem/hmm1_"
        weight_hmm2 = "/content/drive/My Drive/Colab Notebooks/poem/hmm2_"

        train_classifier(poem1, poem2, weight_hmm1, weight_hmm2)

        test(poem1, poem2, weight_hmm1, weight_hmm2)
    else:
        # run on local machine
        poem1 = "../data/nguyen-binh.txt"
        poem2 = "../data/truyen_kieu.txt"

        weight_hmm1 = "../hmm1_"
        weight_hmm2 = "../hmm2_"

        train_classifier(poem1, poem2, weight_hmm1, weight_hmm2)

        test(poem1, poem2, weight_hmm1, weight_hmm2)
