"""
The implementation of hidden markov model using tensorflow for a sequence of discrete observations.

This implementation is simpler than the original version.

The tutorial of the original version can be found here: https://web.stanford.edu/~jurafsky/slp3/A.pdf

"""
import numpy as np
import tensorflow as tf
from graphviz import Digraph


class HMMD_TF:
    def __init__(self, num_of_hidden_states):
        self.num_of_hidden_states = num_of_hidden_states

    def declare_variables(self):
        num_of_vocabularies = len(vocabulary)
        num_of_observations = len(sequence)

        self.tf_observations = tf.compat.v1.placeholder(shape=(num_of_observations,), dtype=tf.int32,
                                                        name='observation')

        self.tf_pi = tf.Variable(initial_value=tf.random.normal(shape=(1, self.num_of_hidden_states)), dtype=tf.float32,
                                 name='pi')
        self.tf_pi = tf.nn.softmax(self.tf_pi)  # this ensures that sum of pi elements is equal to 1

        self.tf_A = tf.Variable(
            initial_value=tf.random.normal(shape=(self.num_of_hidden_states, self.num_of_hidden_states)),
            dtype=tf.float32, name='A')
        self.tf_A = tf.nn.softmax(self.tf_A, axis=1)

        self.tf_B = tf.Variable(initial_value=tf.random.normal(shape=(self.num_of_hidden_states, num_of_vocabularies)),
                                dtype=tf.float32, name='B')
        self.tf_B = tf.nn.softmax(self.tf_B, axis=1)

    def fit(self, observations, vocabulary, iterations=10000):
        self.vocabulary = vocabulary
        self.observations = observations

        self.declare_variables()

        # Define the cost of hidden markov model
        # We need to find A, and B to minimize the cost.
        last_anpha = self.compute_anpha(self.tf_A, self.tf_B, self.tf_pi, self.tf_observations, len(self.observations))
        tf_cost = - tf.math.log(tf.reduce_sum(last_anpha)) # use log to training better
        train_op = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(tf_cost)

        # training
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            for i in range(iterations):
                session.run(train_op, feed_dict={self.tf_observations: observations})

                cost = session.run(tf_cost, feed_dict={self.tf_observations: observations})
                print('iteration ' + str(i) + " : cost = " + str(cost))

            self.A = session.run(self.tf_A, feed_dict={self.tf_observations: observations})
            self.B = session.run(self.tf_B, feed_dict={self.tf_observations: observations})

    def compute_anpha(self, tf_A, tf_B, tf_pi, tf_observations, T):
        last_anpha = tf.math.multiply(tf_pi, tf_B[:, tf_observations[0]])  # 1xN

        for t in range(1, T):
            last_anpha = tf.multiply(tf.matmul(last_anpha, tf_A), tf_B[:, tf_observations[t]])  # 1xN

        return last_anpha

    def draw(self):
        """
        Draw HMM after finishing the training process
        :return:
        """
        dot = Digraph(comment='hmm')

        # create hidden state nodes
        for i in range(self.num_of_hidden_states):
            dot.node('s' + str(i), 'State ' + str(i))

        # create nodes in vocabulary
        for i in range(self.B.shape[1]):
            dot.node('o' + str(i), 'Observation ' + str(i))

        # add weights
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                dot.edge('s' + str(i), 's' + str(j), label=str('%.2f' % self.A[i, j]))

        for i in range(self.B.shape[0]):
            for j in range(self.B.shape[1]):
                dot.edge('s' + str(i), 'o' + str(j), label=str('%.2f' % self.B[i, j]))

        dot.attr(label='Observations = ' + str(sequence) + '\nvocabulary = ' + str(self.vocabulary))
        dot.render('../../graph-output/hmm_graph.gv', view=True)


def read_data(experiment):
    sequence = np.zeros(shape=(len(experiment)))
    vocabulary = dict()

    for trial_idx in range(0, len(experiment)):
        trial = experiment[trial_idx]

        if not trial in vocabulary:
            vocabulary[trial] = len(vocabulary)

        sequence[trial_idx] = vocabulary[trial]

    sequence = sequence.astype(dtype='int')
    return sequence, vocabulary

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    # the size of vocaburary = 2 (i.e., T means tail, H means head)
    sequence, vocabulary = read_data("THTHTHTHTHTHTHTHTHTHTHTHTHTHTHTHTHTHTHTH")

    # train HMM
    hmm = HMMD_TF(num_of_hidden_states=2)  # we can choose a different number of hidden states
    hmm.fit(sequence, vocabulary)
    hmm.draw()