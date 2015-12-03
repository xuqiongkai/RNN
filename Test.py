__author__ = 'xuqiongkai'

import Reader
import Model
import tensorflow as tf
import numpy as np
import rnn_cell

def model_test():
    model = Model.Glove("Models/GloveVec/vectors.6B.100d.small.txt")

    print model.get_vector('china')[0:2]
    print model.get_vector('fdsakjl')
    print model.word_dic.has_key('china')
    print model.word_dic.has_key('fdsa')

def read_file_test():
    pos_data_path = './Dataset/rotten_imdb/plot.tok.gt9.5000'
    neg_data_path = './Dataset/rotten_imdb/plot.tok.gt9.5000'
    pos_data, neg_data = Reader.Reader().read_imdb(pos_data_path, neg_data_path)
    print len(pos_data)
    print len(neg_data)

def rnn_test(sess):
    r = rnn_cell.BasicRNNCell(100)
    state = r.zero_state(1, tf.float32)

    inputs = []
    outputs = []
    states = []
    label_vec = np.zeros((1, 2))
    label_vec[0, 0] = 1
    target = tf.placeholder('float', (1, 2))
    feed_dict = {target: label_vec}
    W = tf.Variable(tf.random_normal([100, 2], stddev=0.01), name="W")
    B = tf.Variable(tf.random_normal([2], stddev=0.01), name="B")

    for t in range(3):
        print 'round', t
        for i in range(5):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            input = tf.placeholder('float', (1, 100))

            word_vec = np.ones((1, 100))
            feed_dict[input] = word_vec
            output_state = r(input, state)

            inputs.append(input)
            (output, state) = output_state
            states.append(state)
            outputs.append(output)

        H = tf.matmul(states[-1], W) + B
        output = tf.nn.softmax(H)
        cost = tf.reduce_sum(-tf.log(output+0.00001)*target)
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

        if i == 1 :
            sess.run(tf.initialize_all_variables())
        print sess.run(output, feed_dict=feed_dict)
        sess.run(optimizer, feed_dict=feed_dict)
        print sess.run(output, feed_dict=feed_dict)


def multi_device_test():
    c = []
    for i in range(10):
        with tf.device('/cpu:0'):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
            c.append(tf.matmul(a, b))
    with tf.device('/cpu:1'):
        sum = tf.add_n(c)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print sess.run(sum)

# model_test()
# read_file_test()

# rnn_test(tf.Session())

multi_device_test()