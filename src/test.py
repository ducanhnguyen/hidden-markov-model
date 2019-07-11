import tensorflow as tf
import numpy as np

def test0():
    x = tf.Variable(initial_value=1, dtype=tf.int32)
    print(x) # x is a Variable
    x = x.assign(1)
    print(x) # x becomes a Tensor with less capacity

def test1():
    a = tf.placeholder(dtype=tf.float32, shape=(3, ))

    x = tf.Variable(initial_value=1, dtype=tf.float32)
    x = x.assign(10) # use this statement, occuring the error 'ValueError: No gradients provided for any variable,'
    # delete this statement, it works!

    cost = tf.reduce_sum(x * a * a)
    train_op = tf.compat.v1.train.AdamOptimizer(1e-2).minimize(cost)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for i in range(100):
            session.run(train_op, feed_dict={a: np.array([1, 2, 3])})

        print(session.run(cost, feed_dict={a: np.array([1, 2, 3])}))
        print(session.run(x))

def test2():
    x = np.zeros(shape=(3, 1))
    for idx, item in enumerate(x):
        print(item)

test2()