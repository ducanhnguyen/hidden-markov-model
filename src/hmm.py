'''
The implementation of hidden markov model based on the tutorial: https://web.stanford.edu/~jurafsky/slp3/A.pdf

I would train a HMM model for an observation

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
- An observation sequence = THTHHHTTHTH
'''
import numpy as np
from graphviz import Digraph


def get_vocaburary(observations_seq):
    vocabulary = dict()

    for observations in observations_seq:
        for observation in observations:
            if observation not in vocabulary:
                vocabulary[observation] = len(vocabulary)

    return vocabulary


def read_data(experiment):
    observations = np.zeros(shape=(len(experiment)))
    vocabulary = dict()

    for trial_idx in range(0, len(experiment)):
        trial = experiment[trial_idx]

        if not trial in vocabulary:
            vocabulary[trial] = len(vocabulary)

        observations[trial_idx] = vocabulary[trial]

    observations = observations.astype(dtype='int')
    return observations, vocabulary


class HMM:

    def fit(self, A, B, pi, observations, vocabulary, iterations=3000):
        """
        Forward-backward algorithm
        :param A: a transaction probability matrix
        :param B: observation likelihoods
        :param pi: an initial probability distribution
        :param observations:
        :param vocabulary:
        :param iterations:
        :return:
        """
        T = len(observations)
        N = A.shape[1]

        for iteration in range(iterations):
            print('\niteration ' + str(iteration))

            # E-step
            anpha = self.compute_anpha(observations, T, N)
            beta = self.compute_beta(observations, T, N)
            zeta = self.compute_zeta(observations, anpha, beta, T, N)
            gama = self.compute_gama(anpha, beta, T, N)

            # M-step
            self.computeA(A, N, T, zeta)
            self.computeB(gama, B, observations, vocabulary, T, N)

            # just for testing
            # P_O_given_lambda should be decreased after iterations
            P_O_given_lambda = self.compute_P_O_given_lambda(anpha, N, T)
            print("P_O_given_lambda = " + str(P_O_given_lambda))

            print('A: ' + str(A))
            print('B: ' + str(B))
            print('pi: ' + str(pi))
            print('vocabulary: ' + str(vocabulary))

    def initialize_hmm_parameters(self, N, vocabulary_size):
        """

        :param N: number of hidden states
        :param vocabulary_size:
        :return:
        """
        # A[i, j]: probability of transforming from state i to state j
        A = np.random.rand(N, N)
        A = A / A.sum(axis=1, keepdims=True)

        # B[i, j]: probability of seeing symbol v_j from state i
        B = np.random.rand(N, vocabulary_size)
        B = B / B.sum(axis=1, keepdims=True)

        pi = np.array([1, 0])
        # pi = np.random.rand(num_hidden_states)
        # pi = pi / pi.sum()
        return A, B, pi

    def compute_P_O_given_lambda(self, anpha, N, T):
        """

        :param anpha:
        :param N: number of hidden states
        :param T: length of an observation
        :return: probability of an observations sequence, given lambda
        """
        P_O_given_lambda = 0
        for i in range(N):
            P_O_given_lambda += anpha[i, T - 1]
        return P_O_given_lambda

    def compute_anpha(self, observations, T, N):

        anpha = np.zeros(shape=(N, T))

        for j in range(N):
            anpha[j, 0] = pi[j] * B[j, observations[0]]

        for t in range(1, T):
            for j in range(N):
                for i in range(N):
                    anpha[j, t] += anpha[i, t - 1] * A[i, j] * B[j, observations[t]]

        return anpha

    def compute_beta(self, observations, T, N):
        beta = np.zeros(shape=(N, T))

        for i in range(N):
            beta[i, T - 1] = 1

        for t in range(T - 2, -1, -1):
            for i in range(N):
                for j in range(N):
                    beta[i, t] += A[i, j] * B[j, observations[t + 1]] * beta[j, t + 1]

        return beta

    def compute_zeta(self, observations, anpha, beta, T, N):
        zeta = np.zeros(shape=(T, N, N))

        for t in range(T - 1):

            P_O_given_lambda = 0
            for j in range(N):
                P_O_given_lambda += anpha[j, t] * beta[j, t]

            for i in range(N):
                for j in range(N):
                    zeta[t, i, j] = anpha[i, t] * A[i, j] * B[j, observations[t + 1]] * beta[j, t + 1]
                    zeta[t, i, j] = zeta[t, i, j] / P_O_given_lambda
        return zeta

    def compute_gama(self, anpha, beta, T, N):
        gama = np.zeros(shape=(N, T))

        for t in range(T):

            P_O_given_lambda = 0
            for j in range(N):
                P_O_given_lambda += anpha[j, t] * beta[j, t]

            for j in range(N):
                gama[j, t] = anpha[j, t] * beta[j, t] / P_O_given_lambda

        return gama

    def computeA(self, A, N, T, zeta):
        for i in range(N):
            for j in range(N):

                numerator = 0
                for t in range(T - 1):
                    numerator += zeta[t, i, j]

                denominator = 0
                for t in range(T - 1):
                    for k in range(N):
                        denominator += zeta[t, i, k]

                A[i, j] = numerator / denominator

        return A

    def computeB(self, gama, B, observations, vocaburary, T, N):
        for j in range(N):

            for k in range(len(vocabulary)):

                numerator = 0
                for t in range(T):
                    if observations[t] == list(vocaburary.values())[k]:
                        numerator += gama[j, t]

                denominator = 0
                for t in range(T):
                    denominator += gama[j, t]

                B[j, list(vocaburary.values())[k]] = numerator / denominator

        return B


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    observations, vocabulary = read_data("TTTTTTTTTTTTTTTTTTHHHHHHHHHHHHHHHHHH")

    # train HMM
    hmm = HMM()
    A, B, pi = hmm.initialize_hmm_parameters(N=2, vocabulary_size=len(vocabulary))
    hmm.fit(A, B, pi, observations, vocabulary)

    # draw HMM
    dot = Digraph(comment='hmm')
    dot.node('s0', 'State 0')
    dot.node('s1', 'State 1')

    dot.node('o0', 'Observation 0' )
    dot.node('o1', 'Observation 1')

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            dot.edge('s' + str(i), 's' + str(j), label=str('%.2f' % A[i, j]))

    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            dot.edge('s' + str(i), 'o' + str(j), label=str('%.2f' % B[i, j]))

    dot.attr(label='Observations = ' + str(observations)+ '\nvocabulary = ' + str(vocabulary))

    dot.render('../../graph-output/hmm_graph.gv', view=True)
