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
import numpy as np
from graphviz import Digraph


def read_data(experiments):
    sequences = []
    vocabulary = dict()

    for exp_list, experiment in enumerate(experiments):
        T = len(experiment)
        observations = np.zeros(shape=(T,))

        for trial_idx in range(0, T):
            trial = experiment[trial_idx]

            if not trial in vocabulary:
                vocabulary[trial] = len(vocabulary)

            observations[trial_idx] = vocabulary[trial]

        observations = observations.astype(dtype='int')
        sequences.append(observations)

    return sequences, vocabulary


class HMMD:
    '''
    Hidden markov model for discrete sequences
    '''

    def __init__(self, num_of_hidden_states):
        self.num_of_hidden_states = num_of_hidden_states  # number of hidden states

    def fit(self, A, B, pi, sequences, vocabulary, iterations=6000):
        """
        Forward-backward algorithm
        :param A: a transaction probability matrix
        :param B: observation likelihoods
        :param pi: an initial probability distribution
        :param sequences:
        :param vocabulary:
        :param iterations:
        :return:
        """
        print('A: ' + str(A))
        print('B: ' + str(B))
        print('pi: ' + str(pi))
        print('vocabulary: ' + str(vocabulary))

        for iteration in range(iterations):
            print('\niteration ' + str(iteration))

            zeta_arr = []
            gama_arr = []
            anpha_arr = []

            for sequence in sequences:
                # E-step
                anpha = self.compute_anpha(sequence, A, B, pi)
                anpha_arr.append(anpha)

                beta = self.compute_beta(sequence, A, B)

                gama = self.compute_gama(anpha, beta)
                gama_arr.append(gama)

                zeta = self.compute_zeta(sequence, anpha, beta, A, B)
                zeta_arr.append(zeta)

            # M-step
            self.compute_A(A, zeta_arr, sequences, anpha_arr)
            self.compute_B(gama_arr, B, sequences, vocabulary, anpha_arr)

            print('A: ' + str(A))
            print('B: ' + str(B))
            print('pi: ' + str(pi))
            print('vocabulary: ' + str(vocabulary))

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
                    anpha[j, t] += anpha[i, t - 1] * A[i, j] * B[j, sequence[t]]

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
                    beta[i, t] += A[i, j] * B[j, sequence[t + 1]] * beta[j, t + 1]

        return beta

    def compute_zeta(self, sequence, anpha, beta, A, B):
        num_of_observations = len(sequence)
        num_of_hidden_states, _ = A.shape
        zeta = np.zeros(shape=(num_of_observations, num_of_hidden_states, num_of_hidden_states))

        for t in range(num_of_observations - 1):

            P_O_given_lambda = 0
            for j in range(num_of_hidden_states):
                P_O_given_lambda += anpha[j, t] * beta[j, t]

            for i in range(num_of_hidden_states):
                for j in range(num_of_hidden_states):
                    zeta[t, i, j] = anpha[i, t] * A[i, j] * B[j, sequence[t + 1]] * beta[j, t + 1]
                    zeta[t, i, j] = zeta[t, i, j] / P_O_given_lambda
        return zeta

    def compute_gama(self, anpha, beta):
        num_of_hidden_states, num_of_observations = anpha.shape

        gama = np.zeros(shape=(num_of_hidden_states, num_of_observations))

        for t in range(num_of_observations):

            P_O_given_lambda = 0
            for j in range(num_of_hidden_states):
                P_O_given_lambda += anpha[j, t] * beta[j, t]

            for j in range(num_of_hidden_states):
                gama[j, t] = anpha[j, t] * beta[j, t] / P_O_given_lambda

        return gama

    def compute_A(self, A, zeta_arr, sequences, anpha_arr, smoothing=1e-5):
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
                A[i, j] = numerator / (denominator + smoothing)

        return A

    def compute_B(self, gama_arr, B, sequences, vocaburary, anpha_arr, smoothing=1e-5):
        num_of_hidden_states, _ = B.shape

        for j in range(num_of_hidden_states):

            for k in range(len(vocabulary)):
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
                        if sequence[t] == list(vocaburary.values())[k]:
                            seq_numerator += gama[j, t]
                    seq_numerator /= (P_O_given_lambda + smoothing)
                    numerator += seq_numerator

                    # compute denominator
                    for t in range(num_of_observations):
                        seq_denominator += gama[j, t]
                    seq_denominator /= (P_O_given_lambda + smoothing)
                    denominator += seq_denominator

                # add smoothing to avoid division by zero
                B[j, list(vocaburary.values())[k]] = numerator / (denominator + smoothing)

        return B

    def find_symbol(self, vocabulary, index_symbol):
        for k, v in vocabulary.items():
            if v == index_symbol:
                return k

    def draw(self, A, B, vocabulary):
        """
        Draw HMM after finishing the training process
        :return:
        """
        dot = Digraph(comment='hmm')

        # create hidden state nodes
        for i in range(self.num_of_hidden_states):
            dot.node('s' + str(i), 'state ' + str(i))

        # create nodes in vocabulary
        for i in range(B.shape[1]):
            dot.node('o' + str(i), 'symbol ' + str(self.find_symbol(vocabulary, i)))

        # add weights
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                dot.edge('s' + str(i), 's' + str(j), label=str('%.2f' % A[i, j]))

        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                dot.edge('s' + str(i), 'o' + str(j), label=str('%.2f' % B[i, j]))

        dot.attr(label='# sequences = ' + str(len(sequences)))
        dot.render('../../graph-output/hmm_graph.gv', view=True)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    # the size of vocabulary = 2 (i.e., T means tail, H means head)
    # two sequences
    sequences, vocabulary = read_data(["TTTTTTTTTTTTTTTTTTTTTTTTTTTTT", "HHHHHHHHHHHHHHHHHHHHHHHHHH"])

    # train HMM
    hmm = HMMD(num_of_hidden_states=2)  # we can choose a different number of hidden states
    A, B, pi = hmm.initialize_hmm_parameters(vocabulary_size=len(vocabulary))
    hmm.fit(A, B, pi, sequences, vocabulary)
    hmm.draw(A, B, vocabulary)

    # just for testing
    for sequence in sequences:
        print('Sequence: ' + str(sequence))
    print('# hidden states = ' + str(hmm.num_of_hidden_states))
