'''
The training time is too long with >=100 lines of poem.
Also, it gives low accuracy.
Maybe HMM is not suitable for classification when the number of symbols is too large.
'''
import sys
from multiprocessing import Process, current_process

import numpy as np
import pandas as pd
import tensorflow as tf
from graphviz import Digraph
import csv
from sklearn.utils import shuffle
from google.colab import drive


@DeprecationWarning
class HMMD_TF:
    def __init__(self):
        pass

    def declare_variables(self, n_hidden_states, vocabulary):
        # create sequences
        tf_sequences = []
        for sequence in self.sequences:
            tf_sequence = tf.compat.v1.placeholder(shape=(len(sequence),), dtype=tf.int32)
            tf_sequences.append(tf_sequence)

        # create initial probability matrix
        tf_pi = tf.Variable(initial_value=tf.random.normal(shape=(1, n_hidden_states)), dtype=tf.float32,
                            name='pi')
        tf_pi = tf.nn.softmax(tf_pi)  # this ensures that sum of pi elements is equal to 1

        # create probability transaction matrix
        tf_A = tf.Variable(
            initial_value=tf.random.normal(shape=(n_hidden_states, n_hidden_states)),
            dtype=tf.float32, name='A')
        tf_A = tf.nn.softmax(tf_A, axis=1)

        # create emission matrix
        n_symbols = len(vocabulary)
        tf_B = tf.Variable(initial_value=tf.random.normal(shape=(n_hidden_states, n_symbols)),
                           dtype=tf.float32, name='B')
        tf_B = tf.nn.softmax(tf_B, axis=1)

        return tf_sequences, tf_pi, tf_A, tf_B

    def fit(self, sequences, vocabulary, matrix_prefix, n_hidden_states, prefix, max_iterations=10000,
            convergence_threshold=1e-10):
        self.process_name = prefix + '[' + current_process().name + ']'
        self.vocabulary = vocabulary
        self.sequences = sequences
        self.n_hidden_states = n_hidden_states

        self.tf_sequences, self.tf_pi, self.tf_A, self.tf_B = self.declare_variables(self.n_hidden_states,
                                                                                     self.vocabulary)

        # Define the cost of hidden markov model
        # We need to find A, and B to minimize the cost.
        tf_cost = 0
        for idx, sequence in enumerate(self.sequences):
            last_anpha = self.compute_anpha(self.tf_A, self.tf_B, self.tf_pi, self.tf_sequences[idx], len(sequence))
            tf_cost += tf.math.log(tf.reduce_sum(last_anpha))

        tf_cost = -tf_cost

        # set adaptive learning rate algorithm
        train_op = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(tf_cost)

        # create feed-dict input
        feed_dict = dict()
        for idx, sequence in enumerate(sequences):
            feed_dict[self.tf_sequences[idx]] = sequences[idx]

        # training
        with tf.compat.v1.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())

            cost_arr = []

            for i in range(max_iterations):
                session.run(train_op, feed_dict)

                cost = session.run(tf_cost, feed_dict)
                cost_arr.append(cost)
                if i%100==0:
                    print(f'{self.process_name} iteration {i}: cost = {cost}')

                # save the current value of hyperparameters to file
                if i % 50 == 0 or i == max_iterations - 1:
                    self.A = session.run(self.tf_A, feed_dict)
                    self.B = session.run(self.tf_B, feed_dict)
                    self.pi = session.run(self.tf_pi, feed_dict)
                    #print(f'{self.process_name} Export hyperparameters to file')
                    self.save(matrix_prefix)

                # check convergence
                if len(cost_arr) >= 2:
                    delta = np.abs(cost_arr[-2] - cost_arr[-1])
                    #print(f'{self.process_name} delta = {delta}')
                    if delta <= convergence_threshold:
                        # converge now
                        print(f'{self.process_name} Converge now. Stop!')
                        break

    def compute_anpha(self, tf_A, tf_B, tf_pi, tf_sequence, T):
        '''
        Compute anpha
        :param tf_A: the tensorflow variable of transaction probability matrix
        :param tf_B: the tensorflow variable of emission matrix
        :param tf_pi: the tensorflow variable of initial probability matric
        :param tf_sequence: the tensorflow variable of 1D array
        :param T: length of the input sequence
        :return: the last anpha
        '''
        last_anpha = tf.math.multiply(tf_pi, tf_B[:, tf_sequence[0]])  # 1xN

        for t in range(1, T):
            last_anpha = tf.multiply(tf.matmul(last_anpha, tf_A), tf_B[:, tf_sequence[t]])  # 1xN

        return last_anpha

    def find_symbol(self, vocabulary, index_symbol):
        for k, v in vocabulary.items():
            if v == index_symbol:
                return k

    def draw(self):
        """
        Draw HMM after finishing the training process
        :return:
        """
        dot = Digraph(comment='hmm')

        # create hidden state nodes
        for i in range(self.n_hidden_states):
            dot.node('s' + str(i), 'state ' + str(i))

        # create nodes in vocabulary
        for i in range(self.B.shape[1]):
            dot.node('o' + str(i), 'symbol ' + str(self.find_symbol(self.vocabulary, i)))

        # add weights
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                dot.edge('s' + str(i), 's' + str(j), label=str(self.A[i, j]))

        for i in range(self.B.shape[0]):
            for j in range(self.B.shape[1]):
                dot.edge('s' + str(i), 'o' + str(j), label=str(self.B[i, j]))

        dot.attr(label='#sequences = ' + str(len(self.sequences)))
        dot.render('../../graph-output/hmm_graph.gv', view=True)

    def save(self, matrix_prefix):
        assert (len(matrix_prefix) > 0)

        assert (self.A.shape[0] == self.A.shape[1])
        with open(str(matrix_prefix) + 'logA.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            [writer.writerow(np.log(r)) for r in self.A]

        with open(str(matrix_prefix) + 'logB.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            [writer.writerow(np.log(r)) for r in self.B]

        assert (self.pi.shape[0] > 0)
        with open(str(matrix_prefix) + 'logpi.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.pi[0])

        with open(str(matrix_prefix) + 'vocabulary.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            for k, v in self.vocabulary.items():
                writer.writerow([k, v])

    def read(self, A_path, B_path, pi_path, vocabulary_path):
        A = np.e ** (pd.read_csv(A_path, header=None).to_numpy())
        B = np.e ** (pd.read_csv(B_path, header=None).to_numpy())
        pi = np.e ** (pd.read_csv(pi_path, header=None).to_numpy())

        v = pd.read_csv(vocabulary_path, header=None)
        vocabulary = dict()
        for idx in range(len(v)):
            key = v.at[idx, 0]
            value = v.at[idx, 1]
            vocabulary[key] = value

        # A and B is stored in log
        self.A = A
        self.B = B
        self.pi = pi
        self.vocabulary = vocabulary
        self.n_hidden_states = A.shape[0]

        return A, B, pi, vocabulary

    def convertSequences2Index(self, X, vocabulary, split_level='CHARACTER'):
        assert (len(vocabulary) > 0)
        X_transform = []

        for x in X:
            # split sequence into fragments
            tokens = []
            if split_level == 'CHARACTER':
                tokens = [character for character in x]
            elif split_level == 'WORD':
                tokens = x.split(' ')
            x_transform = np.zeros(shape=(len(tokens),))

            # convert X to symbol
            for idx, token in enumerate(tokens):

                # ignore if word does not exist in the vocabulary
                for key, value in vocabulary.items():
                    if key == token:
                        x_transform[idx] = value
                        break

            x_transform = x_transform.astype(dtype='int')
            X_transform.append(x_transform)

        return X_transform

    def compute_likelihood(self, X, split_level='CHARACTER'):
        likelihood_arr = []

        X_transform = self.convertSequences2Index(X, self.vocabulary, split_level)

        for x, x_transform in zip(X, X_transform):

            #print(f"x = {x} / x_transform = {x_transform}")

            # compute anpha
            last_anpha = np.multiply(self.pi, self.B[:, x_transform[0]])  # 1xN
            for t in range(1, len(x_transform)):
                last_anpha = np.multiply(np.matmul(last_anpha, self.A), self.B[:, x_transform[t]])  # 1xN

            # likelihood = sum of the last anpha
            likelihood = np.sum(last_anpha)
            likelihood_arr.append(likelihood)

        return likelihood_arr

    def load_data(self, experiments, split_level='CHARACTER'):
        sequences = []
        vocabulary = dict()

        for idx, experiment in enumerate(experiments):
            # define how to split a sequence into fragments
            T = 0
            if split_level == 'CHARACTER':
                T = len(experiment)
            elif split_level == 'WORD':
                experiment = experiment.split(' ')
                T = len(experiment)

            sequence = np.zeros(shape=(T,))

            # iterate over the fragments
            for trial_idx in range(0, T):
                trial = experiment[trial_idx]

                if not trial in vocabulary:
                    vocabulary[trial] = len(vocabulary)

                sequence[trial_idx] = vocabulary[trial]

            sequence = sequence.astype(dtype='int')
            sequences.append(sequence)

        return sequences, vocabulary


def read_data(path, label):
    X = []
    y = []
    poem = open(path).read()

    for line in poem.lower().split("\n"):
        X.append(line)
        y.append(label)

    return X, y


def train_classifier(poem1, poem2):
    limit = 100

    # train model 1
    X1, y1 = read_data(poem1, label=0)
    X1_train = X1[:limit]

    hmm1 = HMMD_TF()
    sequences1, vocabulary1 = hmm1.load_data(X1_train, split_level='WORD')
    process1 = Process(target=hmm1.fit, args=(sequences1, vocabulary1, '/content/drive/My Drive/Colab Notebooks/poem/hmm1_', 20, ''))
    process1.start()

    # train model 2
    X2, y2 = read_data(poem2, label=0)
    X2_train = X2[:limit]

    hmm2 = HMMD_TF()
    sequences2, vocabulary2 = hmm2.load_data(X2_train, split_level='WORD')
    process2 = Process(target=hmm2.fit,
                       args=(sequences2, vocabulary2, '/content/drive/My Drive/Colab Notebooks/poem/hmm2_', 20, '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t'))
    process2.start()

    # join all processes
    process1.join()
    process2.join()


def test(poem1, poem2, base_path):
    # get data
    X1, y1 = read_data(poem1, label=0)
    upper_limit = 1000
    lower_limit = 100
    X1_test = X1[lower_limit:upper_limit]
    y1_test = y1[lower_limit:upper_limit]

    X2, y2 = read_data(poem2, label=1)
    X2_test = X2[lower_limit:upper_limit]
    y2_test = y2[lower_limit:upper_limit]

    X_test = np.concatenate([X1_test, X2_test])
    y_test = np.concatenate([y1_test, y2_test])
    X_test, y_test = shuffle(X_test, y_test)
    #print(X_test)
    # compute likelihood
    hmm1 = HMMD_TF()
    hmm1.read(A_path=base_path+"/hmm1_logA.csv", B_path=base_path+"hmm1_logB.csv", pi_path=base_path+"hmm1_logpi.csv",
              vocabulary_path=base_path+"hmm1_vocabulary.csv")
    likelihood_arr1 = hmm1.compute_likelihood(X_test, split_level='WORD')
    print(f'log likelihood_arr1: {likelihood_arr1}')

    hmm2 = HMMD_TF()
    hmm2.read(A_path=base_path+"hmm2_logA.csv", B_path=base_path+"hmm2_logB.csv", pi_path=base_path+"hmm2_logpi.csv",
              vocabulary_path=base_path+"hmm2_vocabulary.csv")
    likelihood_arr2 = hmm2.compute_likelihood(X_test, split_level='WORD')
    print(f'log likelihood_arr2: {likelihood_arr2}')

    # compute socre
    yhat = np.argmax([likelihood_arr1, likelihood_arr2], axis=0)
    print(f'True ground: {y_test}')
    print(f'Prediction : {yhat}')
    score = np.mean(yhat == y_test)
    print(f'accuracy = {score}')


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    basepath = ''
    GOOGLE_COLAB = False
    if GOOGLE_COLAB:
        # run on colab, but it is still too long
        drive.mount('/content/drive')
        base_path = "/content/drive/My Drive/Colab Notebooks/poem/"
        train_classifier(poem1="/content/drive/My Drive/Colab Notebooks/poem/nguyen-binh.txt",poem2="/content/drive/My Drive/Colab Notebooks/poem/truyen_kieu.txt")
        test(poem1=base_path+"nguyen-binh.txt",poem2=base_path+"truyen_kieu.txt", base_path=base_path)
    else:
        # run on local machine
        basepath='../data/'
        #train_classifier(poem1= "../data/nguyen-binh.txt", poem2 = "../data/truyen_kieu.txt")
        test(poem1= "../data/nguyen-binh.txt", poem2 = "../data/truyen_kieu.txt", base_path="../")
