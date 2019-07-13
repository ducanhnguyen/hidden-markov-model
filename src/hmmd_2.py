'''
The implementation of hidden markov model based on the tutorial https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf

I would train a HMM model for multiple sequences of discrete observations

Definitions:

(1) HMM = (Q, pi, A, B), where

- Q: a set of N hidden states
- N: the number of hidden states
- pi: an initial probability distribution (1xN)
- A: a transaction probability matrix (NxN)
- B: observation likelihoods (NxV)

(2) An observations sequence = o_0, o_1, ..., o_{T-1}; where o_i (0<=t<=T-1) is drawn from a vocabulary.
T: the number of observations in a sequence

For example: Toss a coin
- Q = {0 = head, 1 = tail} (state 0 is head, state 1 is tail)
- N = 2
- A (2x2)
- Vocabulary = {tail, head}
- An example of an observation sequence = THTHHHTTHTH (length=11)
'''
import csv
from multiprocessing import current_process

import numpy as np


class HMMD:
    '''
    Hidden markov model for discrete sequences
    '''

    def __init__(self, num_of_hidden_states):
        self.num_of_hidden_states = num_of_hidden_states  # number of hidden states

    def fit(self, A, B, pi, sequences, vocabulary, max_iterations=6000, converge_threshold=1e-10,
            matrix_prefix='../hmm1_', process_prefix='        ', smoothing=1e-8):
        """
        Forward-backward algorithm
        :param A: a transaction probability matrix
        :param B: observation likelihoods
        :param pi: an initial probability distribution
        :param sequences:
        :param vocabulary:
        :param max_iterations:
        :return:
        """
        assert (converge_threshold >= 0 and len(sequences) >= 1 and A.shape[0] == A.shape[1]
                and pi.shape[0] >= 1 and max_iterations >= 1)

        self.process_name = process_prefix + '[' + current_process().name + ']'

        log_likelihood_arr = []
        for iteration in range(max_iterations):
            print(f'\n{self.process_name} iteration {iteration}')

            # EM algorithm
            zeta_arr = []
            gama_arr = []
            anpha_arr = []

            print(f'{self.process_name} compute anpha, beta, gama, and zeta')
            for sequence in sequences:
                # E-step
                anpha = self.compute_anpha(sequence, A, B, pi)
                anpha_arr.append(anpha)

                beta = self.compute_beta(sequence, A, B)

                gama = self.compute_gama(anpha, beta)
                gama_arr.append(gama)

                zeta = self.compute_zeta(sequence, anpha, beta, A, B)
                zeta_arr.append(zeta)

            # compute likelihood of all sequences before update
            log_likelihood = 0
            for idx, _ in enumerate(sequences):
                log_likelihood += np.log(self.compute_P_O_given_lambda(anpha_arr[idx]) + smoothing)
            print(f'{self.process_name} Log P_O_given_lambda before update= {log_likelihood}')
            log_likelihood_arr.append(log_likelihood)

            # check convergence
            if len(log_likelihood_arr) >= 2 and np.abs(
                    log_likelihood_arr[-2] - log_likelihood_arr[-1]) <= converge_threshold:
                # converge now. no need to update A and B.
                print(f'{self.process_name} Converge now! Stop.')
                break
            else:
                if len(log_likelihood_arr) >= 2:
                    print(str(self.process_name) + ' delta likelihood = ' + str(
                        np.abs(log_likelihood_arr[-2] - log_likelihood_arr[-1])))

                # M-step
                print(f'{self.process_name} re-estimate A')
                self.compute_A(A, zeta_arr, sequences, anpha_arr)

                print(f'{self.process_name} re-estimate B')
                self.compute_B(gama_arr, B, sequences, vocabulary, anpha_arr)

            # export matrix to file after iterations
            if len(matrix_prefix) > 0:
                self.A = A
                self.B = B
                self.pi = pi
                self.vocabulary = vocabulary

                self.save(matrix_prefix)

    def initialize_hmm_parameters(self, vocabulary_size):
        """

        :param N: number of hidden states
        :param vocabulary_size:
        :return:
        """

        # A[i, j]: probability of transforming from state i to state j
        A = np.random.rand(self.num_of_hidden_states, self.num_of_hidden_states)
        A = A / A.sum(axis=1, keepdims=True)

        # B[i, j]: probability of seeing symbol v_j from state i
        B = np.random.rand(self.num_of_hidden_states, vocabulary_size)
        B = B / B.sum(axis=1, keepdims=True)

        pi = np.random.rand(self.num_of_hidden_states)
        pi = pi / pi.sum()
        return A, B, pi

    def compute_P_O_given_lambda(self, anpha):
        """

        :param anpha:
        :param num_of_hidden_states: number of hidden states
        :param num_of_observations: length of an observation
        :return: probability of an observations sequence, given lambda
        """
        P_O_given_lambda = 0
        num_of_hidden_states, num_of_observations = anpha.shape

        for i in range(num_of_hidden_states):
            P_O_given_lambda += anpha[i, num_of_observations - 1]

        return P_O_given_lambda

    def compute_anpha(self, sequence, A, B, pi):
        num_of_observations = len(sequence)
        num_of_hidden_states, _ = A.shape
        anpha = np.zeros(shape=(num_of_hidden_states, num_of_observations))

        for j in range(num_of_hidden_states):
            anpha[j, 0] = pi[j] * B[j, sequence[0]]

        for t in range(1, num_of_observations):
            for j in range(num_of_hidden_states):
                for i in range(num_of_hidden_states):
                    log = np.log(anpha[i, t - 1]) + np.log(A[i, j]) + np.log(B[j, sequence[t]])
                    anpha[j, t] += np.e ** log

        return anpha

    def compute_beta(self, sequence, A, B):
        num_of_observations = len(sequence)
        num_of_hidden_states, _ = A.shape
        beta = np.zeros(shape=(num_of_hidden_states, num_of_observations))

        for i in range(num_of_hidden_states):
            beta[i, num_of_observations - 1] = 1

        for t in range(num_of_observations - 2, -1, -1):
            for i in range(num_of_hidden_states):
                for j in range(num_of_hidden_states):
                    log = np.log(A[i, j]) + np.log(B[j, sequence[t + 1]]) + np.log(beta[j, t + 1])
                    beta[i, t] += np.e ** log

        return beta

    def compute_zeta(self, sequence, anpha, beta, A, B, smoothing=1e-8):
        num_of_observations = len(sequence)
        num_of_hidden_states, _ = A.shape
        zeta = np.zeros(shape=(num_of_observations, num_of_hidden_states, num_of_hidden_states))

        for t in range(num_of_observations - 1):

            P_O_given_lambda = 0
            for j in range(num_of_hidden_states):
                P_O_given_lambda += anpha[j, t] * beta[j, t]

            for i in range(num_of_hidden_states):
                for j in range(num_of_hidden_states):
                    log = np.log(anpha[i, t]) + np.log(A[i, j]) + np.log(B[j, sequence[t + 1]]) + np.log(beta[j, t + 1])
                    zeta[t, i, j] = np.e ** log / (P_O_given_lambda + smoothing)
        return zeta

    def compute_gama(self, anpha, beta, smoothing=1e-5):
        num_of_hidden_states, num_of_observations = anpha.shape

        gama = np.zeros(shape=(num_of_hidden_states, num_of_observations))

        for t in range(num_of_observations):

            P_O_given_lambda = 0
            for j in range(num_of_hidden_states):
                P_O_given_lambda += anpha[j, t] * beta[j, t]

            for j in range(num_of_hidden_states):
                gama[j, t] = anpha[j, t] * beta[j, t] / (P_O_given_lambda + smoothing)

        return gama

    def compute_A(self, A, zeta_arr, sequences, anpha_arr, smoothing=1e-8):
        num_of_hidden_states, _ = A.shape

        for i in range(num_of_hidden_states):
            for j in range(num_of_hidden_states):
                numerator = 0
                denominator = 0

                # iterate over sequences of observations
                for idx_seq, sequence in enumerate(sequences):
                    zeta = zeta_arr[idx_seq]
                    seq_numerator = 0
                    seq_denominator = 0

                    # P(O|lamba) can be thought as a weight of a sequence
                    P_O_given_lambda = self.compute_P_O_given_lambda(anpha_arr[idx_seq])

                    # compute numerator
                    num_of_observations = len(sequence)
                    for t in range(num_of_observations - 1):
                        seq_numerator += zeta[t, i, j]
                    seq_numerator /= (P_O_given_lambda + smoothing)
                    numerator += seq_numerator

                    # compute denominator
                    for t in range(num_of_observations - 1):
                        for k in range(num_of_hidden_states):
                            seq_denominator += zeta[t, i, k]
                    seq_denominator /= (P_O_given_lambda + smoothing)
                    denominator += seq_denominator

                # add smoothing to avoid division by zero
                A[i, j] = (numerator + smoothing) / (denominator + smoothing)

        return A

    def compute_B_element(self, B, vocabulary, gama_arr, sequences, anpha_arr, smoothing, hidden_state_id):
        # print(f'{self.process_name} compute B element')
        for symbol_id in range(len(vocabulary)):
            # print(f'{self.process_name} compute B: hidden state {hidden_state_id} / vocabulary {symbol_id} : {len(vocabulary)}')

            numerator = 0
            denominator = 0

            for idx_seq, sequence in enumerate(sequences):

                seq_numerator = 0
                seq_denominator = 0
                gama = gama_arr[idx_seq]

                # P(O|lamba) can be thought as a weight of a sequence
                P_O_given_lambda = self.compute_P_O_given_lambda(anpha_arr[idx_seq])

                # compute numerator
                num_of_observations = len(sequence)
                for t in range(num_of_observations):
                    if sequence[t] == list(vocabulary.values())[symbol_id]:
                        seq_numerator += gama[hidden_state_id, t]
                seq_numerator /= (P_O_given_lambda + smoothing)
                numerator += seq_numerator

                # compute denominator
                for t in range(num_of_observations):
                    seq_denominator += gama[hidden_state_id, t]
                seq_denominator /= (P_O_given_lambda + smoothing)
                denominator += seq_denominator

            # add smoothing to avoid division by zero
            B[hidden_state_id, list(vocabulary.values())[symbol_id]] = (numerator + smoothing) / (
                        denominator + smoothing)

    def compute_B(self, gama_arr, B, sequences, vocabulary, anpha_arr, smoothing=1e-8):
        num_of_hidden_states, _ = B.shape

        # processes = []
        for hidden_state_id in range(num_of_hidden_states):
            print(f'{self.process_name} Compute B with hidden state {hidden_state_id} / {num_of_hidden_states}')
            self.compute_B_element(B, vocabulary, gama_arr, sequences, anpha_arr, smoothing, hidden_state_id)

            # put in multithread to accelerate the training time
            '''
            process = Process(target=self.compute_B_element,
                              args=(B, vocabulary, gama_arr, sequences, anpha_arr, smoothing, hidden_state_id))
            processes.append(process)
            process.start()
            '''

        # for process in processes:
        #    process.start()

        # for process in processes:
        #    process.join()

        return B

    def convertSequences2Index(self, X, vocabulary):
        assert (len(vocabulary) > 0)
        X_transform = []
        for x in X:
            tokens = x.split(" ")

            # convert X to symbol
            x_transform = np.zeros(shape=(len(tokens),))
            for idx, token in enumerate(tokens):

                # ignore if word does not exist in the vocabulary
                for key, value in vocabulary.items():
                    if key == token:
                        x_transform[idx] = value
                        break

            x_transform = x_transform.astype(dtype='int')
            X_transform.append(x_transform)

        return X_transform

    def compute_loglikelihood(self, X):
        likelihood_arr = []

        X_transform = self.convertSequences2Index(X, self.vocabulary)

        for x, x_transform in zip(X, X_transform):
            # compute likelihood
            print(f"x = {x} / x_transform = {x_transform}")
            anpha = self.compute_anpha(x_transform, self.A, self.B, self.pi)
            likelihood = self.compute_P_O_given_lambda(anpha)
            likelihood_arr.append(np.log(likelihood))

        return likelihood_arr

    def save(self, matrix_prefix):
        print(f'{self.process_name} Export to file')

        assert (self.A.shape[0] == self.A.shape[1])
        with open(str(matrix_prefix) + 'A.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            [writer.writerow(np.log(r)) for r in self.A]

        with open(str(matrix_prefix) + 'B.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            [writer.writerow(np.log(r)) for r in self.B]

        assert (self.pi.shape[0] > 0)
        with open(str(matrix_prefix) + 'pi.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.pi)

        with open(str(matrix_prefix) + 'vocabulary.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            for k, v in self.vocabulary.items():
                writer.writerow([k, v])
