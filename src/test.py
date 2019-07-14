import csv

import numpy as np
import tensorflow as tf
import pandas as pd

def test0():
    x = tf.Variable(initial_value=1, dtype=tf.int32)
    print(x)  # x is a Variable
    x = x.assign(1)
    print(x)  # x becomes a Tensor with less capacity


def test1():
    a = tf.placeholder(dtype=tf.float32, shape=(3,))

    x = tf.Variable(initial_value=1, dtype=tf.float32)
    x = x.assign(10)  # use this statement, occuring the error 'ValueError: No gradients provided for any variable,'
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


def test3():
    x = np.array([1, 12, 3])
    y = np.array([4, 5, 6])
    z = [x, y]
    max = np.argmax(z, axis=0)
    print(max)


def test4():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # write
    with open('../matrix_test.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in a]

    # read it

    data = pd.read_csv('../matrix_test.csv', header=None)
    data = data.to_numpy()
    print(data.shape)
    print(data)


def test5():
    x = dict()
    x['anh'] = 0
    x['toi'] = 1

    # write
    with open('../matrix_test.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for k, v in x.items():
            writer.writerow([k, v])

    # read
    data = pd.read_csv('../matrix_test.csv', header=None)
    for row in range(len(data)):
        a = data.at[row,0]
        b = data.at[row, 1]
        print(f"{a} {b}")

def test6():
    x = 4
    y = np.log(x)
    print(y)
    z = np.e **y
    print(z)

def test7():
    x1 = np.array([[1, 2], [3, 4]])
    x2 = np.array([[1, 20], [30, 4]])
    y = np.mean([x1==x2], axis=None)
    z = np.concatenate(x1)
    print(z)
    print(y)

test7()
