__author__ = 'xuqiongkai'

import tensorflow as tf

def sequence_rnn(rnn_cell, word_list, word_model, first_train=True):
    feed_dict = {}
    state = rnn_cell.zero_state(1, tf.float32)
    counter = 0

    outputs = []
    states = []
    inputs = []
    for word in word_list:
        word_vec = word_model.get_vector(word)
        if word_vec is not None:
            counter += 1
            if counter > 1 or first_train==False:

                tf.get_variable_scope().reuse_variables()
            input = tf.placeholder('float', (1, word_model.dim))
            word_vec = word_vec.reshape((1, word_model.dim))
            feed_dict[input] = word_vec
            output_state = rnn_cell(input, state)
            #inputs.append(input)
            (output, state) = output_state
            states.append(state)
            outputs.append(output)

    return outputs[-1], feed_dict
