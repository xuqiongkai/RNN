__author__ = 'xuqiongkai'

import tensorflow as tf

def sequence_rnn_pad(rnn_cell, input_dim, length=50, first_train=True):
    state = rnn_cell.zero_state(1, tf.float32)
    outputs = []
    inputs = []
    states = []
    flags = []
    for i in range(length):
        if i > 0 or first_train == False:
            tf.get_variable_scope().reuse_variables()
        input = tf.placeholder('float', (1, input_dim))
        inputs.append(input)

        output_state = rnn_cell(input, state)
        (output, state) = output_state

        flag = tf.placeholder('float', (1, rnn_cell.state_size))
        state = flag * state
        flags.append(flag)
        # flag = tf.placeholder(tf.types.float32)
        # flags.append(flag)
        # state = flag * state

        states.append(state)
        outputs.append(output)

    # for i in range(length):
    #     flag = tf.Variable(0)
    #     flags.append(flag)
    #     states[i] = flag * states[i]

    return inputs, outputs, states, flags

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

    return states[-1], feed_dict
