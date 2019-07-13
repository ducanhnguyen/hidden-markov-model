"""
The implementation of hidden markov model using tensorflow for multiple sequences of discrete observations.

This implementation is simpler than the original version.

The tutorial of the original version can be found here: https://web.stanford.edu/~jurafsky/slp3/A.pdf

"""
import csv
from multiprocessing import current_process

import numpy as np
import pandas as pd
import tensorflow as tf
from graphviz import Digraph


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

    def fit(self, sequences, vocabulary, matrix_prefix, n_hidden_states, prefix, max_iterations=20000,
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
                print(f'{self.process_name} iteration {i}: cost = {cost}')

                # save the current value of hyperparameters to file
                if i % 50 == 0 or i == max_iterations - 1:
                    self.A = session.run(self.tf_A, feed_dict)
                    self.B = session.run(self.tf_B, feed_dict)
                    self.pi = session.run(self.tf_pi, feed_dict)
                    print(f'{self.process_name} Export hyperparameters to file')
                    self.save(matrix_prefix)

                # check convergence
                if len(cost_arr) >= 2:
                    delta = np.abs(cost_arr[-2] - cost_arr[-1])
                    print(f'{self.process_name} delta = {delta}')
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

            print(f"x = {x} / x_transform = {x_transform}")

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


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    # Step 1. train HMM
    '''
    # The value of hyperparameters (e.g., A, B, pi) will be stored in external files for further usages.
    hmm = HMMD_TF()  # we can choose a different number of hidden states
    # the size of vocaburary = 2 (i.e., T means tail, H means head)
    experiments = ["T T T T T T T T T T T T T T T T T T T T T T T T T T T T H T T", "H H H H H H H H H H H H H H H H H H T H H H H H H"]
    sequences, vocabulary = hmm.load_data(experiments, split_level='WORD')
    hmm.fit(sequences, vocabulary, n_hidden_states=2, matrix_prefix='../hmm_')
    hmm.draw()
    '''

    # Step 2. test HMM
    # You can load hyperparameter files without training anymore.
    print('Test. Read weights from file')
    hmm2 = HMMD_TF()  # no need to specify the number of hidden states
    hmm2.read(A_path='../hmm_logA.csv', B_path='../hmm_logB.csv', pi_path='../hmm_logpi.csv',
              vocabulary_path='../hmm_vocabulary.csv')
    Xtest = ["T H T H T H T H T H T H T H T H",
             "T T T T T T T T T T T T T T T T",
             "H H H H H H H H H H H H H H H H",
             "H H H T H H H H H H T H H H H H",
             "H H H H H H H H H H T H H H H H"]
    likelihood_arr = hmm2.compute_likelihood(Xtest, split_level='WORD')
    for sequence, likelihood in zip(Xtest, likelihood_arr):
        print(f'Test {sequence} : probability = {likelihood} (log of probability = {np.log(likelihood)})')
