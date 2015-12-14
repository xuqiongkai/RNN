__author__ = 'xuqiongkai'
import tensorflow as tf
import numpy as np
import rnn_cell
import rnn
# from tensorflow.models.rnn import  rnn

class PadClassifier(object):
    def __init__(self, word_model, class_num=2, unit_num=100, learn_rate=0.005):
        self.word_model = word_model
        self.class_num = class_num
        self.rnn_cell = rnn_cell.BasicLSTMCell(unit_num)
        self.target = tf.placeholder('float', (1, class_num))
        self.W = tf.Variable(tf.random_normal([self.rnn_cell.state_size, class_num], stddev=0.01), name="W")
        self.B = tf.Variable(tf.random_normal([class_num], stddev=0.01), name="B")
        self.seq_length = 40

        self.inputs, self.outputs, self.states, self.flags = rnn.sequence_rnn_pad(self.rnn_cell, word_model.dim, self.seq_length, True)

        #self.S = (self.states[-1] + self.states[-2] + self.states[-3]) / 3
        #self.H = tf.matmul(self.S, self.W) + self.B
        self.H = tf.matmul(self.states[-1], self.W) + self.B
        self.output = tf.nn.softmax(self.H)

        self.cost = tf.reduce_sum(-tf.log(self.output+0.00001)*self.target)
        self.optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(self.cost)
        #self.optimizer = tf.train.MomentumOptimizer(0.5, 0.1).minimize(self.cost)

    def test_sample(self, word_list, sess):
        feed_dict = {}
        counter = 0
        for i in range(self.seq_length):
            feed_dict[self.inputs[i]] = np.zeros((1, self.word_model.dim))
            feed_dict[self.flags[i]] = np.zeros((1, self.rnn_cell.state_size))

        for word in word_list[::-1]:
            if counter >= self.seq_length:
                break
            word_vec = self.word_model.get_vector(word)
            if word_vec is not None:
                counter += 1
                word_vec = word_vec.reshape((1, self.word_model.dim))
                feed_dict[self.inputs[self.seq_length - counter]] = word_vec
                feed_dict[self.flags[self.seq_length - counter]] = np.ones((1, self.rnn_cell.state_size))

        return sess.run(self.output, feed_dict=feed_dict)


    def train_sample(self, word_list, label, sess):
        feed_dict = {}
        # init label vector
        label_vec = np.zeros((1, self.class_num))
        label_vec[0, label] = 1
        feed_dict[self.target] = label_vec

        counter = 0
        for i in range(self.seq_length):
            feed_dict[self.inputs[i]] = np.zeros((1, self.word_model.dim))
            feed_dict[self.flags[i]] = np.zeros((1, self.rnn_cell.state_size))

        for word in word_list[::-1]:
            if counter >= self.seq_length:
                break
            word_vec = self.word_model.get_vector(word)
            if word_vec is not None:
                counter += 1
                word_vec = word_vec.reshape((1, self.word_model.dim))
                feed_dict[self.inputs[self.seq_length - counter]] = word_vec
                feed_dict[self.flags[self.seq_length - counter]] = np.ones((1, self.rnn_cell.state_size))

        #print sess.run(self.output, feed_dict=feed_dict)
        sess.run(self.optimizer, feed_dict=feed_dict)
        #print sess.run(self.output, feed_dict=feed_dict)


class ChainClassifier(object):
    def __init__(self, word_model, class_num=2, unit_num=100, learn_rate=0.005):
        self.word_model = word_model
        self.class_num = class_num
        self.unit_num = unit_num
        self.rnn_cell = rnn_cell.BasicLSTMCell(unit_num)
        self.W = tf.Variable(tf.random_normal([self.rnn_cell.state_size, class_num], stddev=0.01), name="W")
        self.B = tf.Variable(tf.random_normal([class_num], stddev=0.01), name="B")
        self.target = tf.placeholder('float', (1, self.class_num))
        self.first_train = True
        self.learn_rate = learn_rate

    def test_sample(self, word_list, sess):
        # generate sequence structure and input dictionary
        state, feed_dict = rnn.sequence_rnn(self.rnn_cell, word_list, self.word_model, self.first_train)

        H = tf.matmul(state, self.W) + self.B
        output = tf.nn.softmax(H)
        return sess.run(output, feed_dict=feed_dict)


    def train_sample(self, word_list, label, sess):
        # generate sequence structure and input dictionary
        state, feed_dict = rnn.sequence_rnn(self.rnn_cell, word_list, self.word_model, self.first_train)

        # init label vector
        label_vec = np.zeros((1, self.class_num))
        label_vec[0, label] = 1
        feed_dict[self.target] = label_vec

        # latest layer (softmax)
        #with tf.device('/cpu:0'):
        H = tf.matmul(state, self.W) + self.B
        output = tf.nn.softmax(H)
        cost = tf.reduce_sum(-tf.log(output+0.00001)*self.target)
        optimizer = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(cost)

        if self.first_train:
            sess.run(tf.initialize_all_variables())
            self.first_train = False
        #print sess.run(output, feed_dict=feed_dict)
        sess.run(optimizer, feed_dict=feed_dict)
        #print sess.run(output, feed_dict=feed_dict)


# remove later
    def train_sample_save(self, word_list, label, sess):
        outputs = []
        states = []
        inputs = []

        label_vec = np.zeros((1, self.class_num))
        label_vec[0, label] = 1
        feed_dict = {self.target: label_vec}

        state = self.rnn_cell.zero_state(1, tf.float32)
        counter = 0
        for word in word_list:
            word_vec = self.word_model.get_vector(word)
            if word_vec is not None:
                counter += 1
                if counter > 1 or self.first_train==False:
                    #break
                    tf.get_variable_scope().reuse_variables()
                input = tf.placeholder('float', (1, self.word_model.dim))

                word_vec = word_vec.reshape((1, self.word_model.dim))
                feed_dict[input] = word_vec
                output_state = self.rnn_cell(input, state)

                inputs.append(input)
                (output, state) = output_state
                states.append(state)
                outputs.append(output)

        if self.first_train or True:
            sess.run(tf.initialize_all_variables())
            self.first_train = False
        #print states[-1]
        H = tf.matmul(states[-1], self.W) + self.B
        output = tf.nn.softmax(H)
        cost = tf.reduce_sum(-tf.log(output+0.00001)*self.target)
        optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

        #print sess.run(output, feed_dict=feed_dict)
        sess.run(optimizer, feed_dict=feed_dict)
        print sess.run(output, feed_dict=feed_dict)






class PlainClassifier(object):
    def __init__(self, word_model, class_num=2):
        self.word_model = word_model
        self.class_num = class_num
        self.input = tf.placeholder('float', (None, word_model.dim))
        self.target = tf.placeholder('float', (None, class_num))
        self.W = tf.Variable(tf.random_normal([word_model.dim, class_num], stddev=0.01), name="W")
        self.B = tf.Variable(tf.random_normal([class_num], stddev=0.01), name="B")

        self.H = tf.matmul(self.input, self.W) + self.B
        self.output = tf.nn.softmax(self.H)

        self.cost = tf.reduce_sum(-tf.log(self.output+0.00001)*self.target)
        #self.optimizer = tf.train.GradientDescentOptimizer(0.00002).minimize(self.cost)
        self.optimizer = tf.train.AdagradOptimizer(0.0005).minimize(self.cost)

    def get_doc_vector(self, word_List):
        count = 0
        doc_vec = np.zeros(self.word_model.dim)
        for word in word_List:
            word_vec = self.word_model.get_vector(word)
            if word_vec is not None:
                doc_vec += word_vec
                count += count
            if count > 0:
                doc_vec /= count
        return doc_vec

    def test_sample(self, word_List, sess):
        doc_vec = self.get_doc_vector(word_List)
        doc_vec = doc_vec.reshape((1, self.word_model.dim))
        feed_dict = {self.input: doc_vec}
        return sess.run(self.output, feed_dict=feed_dict)

    def train_sample(self, word_List, label, sess):
        doc_vec = self.get_doc_vector(word_List)
        doc_vec = doc_vec.reshape((1, self.word_model.dim))
        label_vec = np.zeros((1, self.class_num))
        label_vec[0, label] = 1
        feed_dict = {self.input: doc_vec, self.target: label_vec}
        #print sess.run(self.output, feed_dict), sess.run(-tf.log(self.output), feed_dict), sess.run(self.cost, feed_dict)
        sess.run(self.optimizer, feed_dict=feed_dict)


