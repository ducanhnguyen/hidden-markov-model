'''
This is a POS Tagging Technique using HMM.
We do not need to train HMM anymore but we use a simpler approach. Hidden state is pos tagger.

Explanation of pos tag can be found here: https://pythonprogramming.net/part-of-speech-tagging-nltk-tutorial/
Dataset: https://www.clips.uantwerpen.be/conll2000/chunking/

Result:
- Training f1-score = 0.9241242452935616
- Test f1-score = 0.873736957062177
'''
import csv

import numpy as np
from sklearn.metrics import f1_score


class POS:
    '''
    Implementation of pos tagger using HMM.
    '''
    def __init__(self):
        self.smoothing = 1e-5
        pass

    def read_data(self, train_file):
        sequences = []
        hidden_state_chains = []

        sequence = []
        hidden_state_chain = []
        with open(file=train_file) as f:

            lines = f.read().split('\n')

            for idx, line in enumerate(lines):

                if len(line) == 0:
                    # ignore the break line
                    # this is the separation between two sentences
                    if len(sequence) > 0 and len(hidden_state_chain) > 0:
                        sequences.append(sequence)
                        hidden_state_chains.append(hidden_state_chain)

                    sequence = []
                    hidden_state_chain = []
                else:
                    tokens = line.split(' ')
                    sequence.append(tokens[0])
                    hidden_state_chain.append(tokens[1])

        return sequences, hidden_state_chains

    def create_hidden_states(self, hidden_state_chains):
        hidden_states_set = set()

        for chain in hidden_state_chains:
            for token in chain:
                hidden_states_set.add(token)

        return list(hidden_states_set)

    def create_A(self, hidden_states, hidden_state_chains, state2index):
        n_hidden_states = len(hidden_states)
        A = np.zeros(shape=(n_hidden_states, n_hidden_states))

        for idx, chain in enumerate(hidden_state_chains):
            for idx2 in range(len(chain) - 1):
                state_i = state2index[chain[idx2]]
                state_j = state2index[chain[idx2 + 1]]
                A[state_i, state_j] += 1  # the number of transactions from state_i -> state_j

        A += self.smoothing  # avoid division by zero
        A /= np.sum(A, axis=1, keepdims=True)  # row sums up to 1
        return A

    def create_B(self, hidden_state_chains, sequences, hidden_states, vocabulary, state2index, symbol2index):
        n_hidden_states = len(hidden_states)
        n_symbols = len(vocabulary)
        B = np.zeros(shape=(n_hidden_states, n_symbols))

        for idx, chain in enumerate(hidden_state_chains):
            for idx2 in range(len(chain)):
                state_i = state2index[chain[idx2]]

                symbol = sequences[idx][idx2].lower()
                symbol_i = symbol2index[symbol]

                B[state_i, symbol_i] += 1  # the number of transactions from state_i -> state_j

        B += self.smoothing  # avoid division by zero
        B /= np.sum(B, axis=1, keepdims=True)  # row sums up to 1
        return B

    def create_pi(self, hidden_state_chains, state2index):
        pi = np.zeros(shape=(len(state2index),))

        for chain in hidden_state_chains:
            first_token = chain[0]
            pi[state2index[first_token]] += 1

        pi /= np.sum(pi)
        return pi

    def create_vocabulary(self, sequences):
        vocabulary = set()

        for sequence in sequences:
            for token in sequence:
                vocabulary.add(token.lower())
        return list(vocabulary)

    def create_state2index(self, hidden_states):
        state2index = dict()

        for idx, state in enumerate(hidden_states):
            state2index[state] = idx
        return state2index

    def create_symbol2index(self, vocabulary):
        symbol2index = dict()

        for idx, symbol in enumerate(vocabulary):
            symbol2index[symbol] = idx
        return symbol2index

    def decoding(self, X, state2index, symbol2index, pi, A, B, spilit_level='WORD'):
        '''
        Find the most likely hidden states
        :param X: list of phrases
        :param state2index:
        :param symbol2index:
        :param state2index: a dictionary
        :param symbol2index: a dictionary
        :param pi: initial probability matrix
        :param A: probability transaction matrix
        :param B: emision matrix
        :param spilit_level: define how to split phrases.
        :return: the most likely hidden states
        '''
        assert (spilit_level == 'WORD' or spilit_level == 'NONE')

        most_likely_chains = []

        for idx, x in enumerate(X):
            if idx % 100 == 0 or idx == len(X) - 1:
                print(f'Decoding {idx} / {len(X)}')

            # create observations
            observations = []
            if spilit_level == 'WORD':
                observations = x.lower().split(' ')
            elif spilit_level == 'NONE':
                observations = [word.lower() for word in x]

            # compute beta matrix
            T = len(observations)
            N = len(state2index)

            beta = np.zeros(shape=(N, T))

            if observations[0] in symbol2index:
                current_observation = symbol2index[observations[0]]
                # beta[:, 0] = np.log(pi) + np.log(B[:, current_observation])  # B: (hidden states, symbols)
                beta[:, 0] = pi * B[:, current_observation]  # B: (hidden states, symbols)
            else:
                # beta[:, 0] = np.log(pi) + np.log(self.smoothing)
                beta[:, 0] = pi * self.smoothing

            for t in range(1, T):
                for n1 in range(N):

                    prob = []
                    for n2 in range(N):
                        if observations[t] in symbol2index:
                            # p = np.log(beta[n2, t - 1]) + np.log(A[n2, n1]) + np.log(B[n1, symbol2index[observations[t]]])
                            p = beta[n2, t - 1] * A[n2, n1] * B[n1, symbol2index[observations[t]]]
                        else:
                            # a word in the training set but not in test set
                            # p = np.log(self.smoothing)
                            p = self.smoothing

                        prob.append(p)

                    # max index
                    beta[n1, t] = np.max(prob)

            # find the most likely hidden states
            most_likely_states = []
            for t in range(0, T):
                max = np.argmax(beta[:, t])

                for k, v in state2index.items():
                    if v == max:
                        most_likely_states.append(k)
                        break
            most_likely_chains.append(most_likely_states)

        return most_likely_chains

    def f1_score(self, X, Y, state2index, symbol2index, pi, A, B, spilit_level='WORD'):
        '''
        Compute f1 score of the given phrases
        :param X: list of phrases
        :param Y: the target
        :param state2index: a dictionary
        :param symbol2index: a dictionary
        :param pi: initial probability matrix
        :param A: probability transaction matrix
        :param B: emision matrix
        :param spilit_level: define how to split phrases.
        :return:
        '''
        assert (spilit_level == 'WORD' or spilit_level == 'NONE')
        print(f'# sequences = {len(X)}')

        print(f'Find the most likely pos tag with the given phrases')
        Yhat = self.decoding(X, state2index, symbol2index, pi, A, B, spilit_level)

        print(f'Computing f1 score')
        Y = np.concatenate(Y)  # flatten
        Yhat = np.concatenate(Yhat)  # flatten
        score = f1_score(Y, Yhat, average=None).mean()
        return score

    def export(self, matrix, file):
        '''
        Export matrix to file
        :param matrix: 1D matrix or 2D matrix
        :param file: the location
        :return:
        '''
        with open(file, 'w') as csvfile:
            writer = csv.writer(csvfile)

            if matrix.ndim == 2:  # 2d
                [writer.writerow(r) for r in matrix]
            elif matrix.ndim == 1:  # 1d
                writer.writerow(matrix)

        return True


def train():
    pos = POS()
    sequences, hidden_state_chains = pos.read_data(train_file='../data/chunking/train.txt')

    hidden_states = pos.create_hidden_states(hidden_state_chains)
    state2index = pos.create_state2index(hidden_states)

    A = pos.create_A(hidden_states, hidden_state_chains, state2index)

    vocabulary = pos.create_vocabulary(sequences)
    symbol2index = pos.create_symbol2index(vocabulary)

    B = pos.create_B(hidden_state_chains, sequences, hidden_states, vocabulary, state2index, symbol2index)

    pi = pos.create_pi(hidden_state_chains, state2index)

    return A, B, pi, symbol2index, state2index, sequences, hidden_state_chains


if __name__ == '__main__':
    A, B, pi, symbol2index, state2index, train_sequence, train_hidden_state_chains = train()

    # training set
    f1 = POS().f1_score(train_sequence[:None], train_hidden_state_chains[:None], state2index, symbol2index, pi, A, B,
                        'NONE')
    print(f'Training f1-score = {f1}')

    # test set
    pos = POS()
    test_sequences, test_hidden_state_chains = pos.read_data(train_file='../data/chunking/test.txt')
    f1 = pos.f1_score(test_sequences[:None], test_hidden_state_chains[:None], state2index, symbol2index, pi, A, B,
                      'NONE')
    print(f'Test f1-score = {f1}')

    # test with a phrase
    X = ['i love you so much']
    # expected: [['PRP', 'VBP', 'PRP', 'RB', 'JJ']]
    tag = POS().decoding(X, state2index, symbol2index, pi, A, B, spilit_level='WORD')
    print(f'tag of phrase {X} = {tag}')
