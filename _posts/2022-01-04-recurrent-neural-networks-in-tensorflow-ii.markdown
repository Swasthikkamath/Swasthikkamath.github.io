---
layout: post
title:  "Recurrent Neural Networks in Tensorflow II"
date:   2022-01-04 11:28:34 +0530
categories: jekyll tensorflow neuralNetworks
---

This is the second in a series of posts about recurrent neural networks in Tensorflow. The first post lives [here](https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html). In this post, we will build upon our vanilla RNN by learning how to use Tensorflow’s scan and dynamic_rnn models, upgrading the RNN cell and stacking multiple RNNs, and adding dropout and layer normalization. We will then use our upgraded RNN to generate some text, character by character.

**Note 3/14/2017:** This tutorial is quite a bit deprecated by changes to the TF api. Leaving it up since it may still be useful, and most changes to the API are cosmetic (biggest change is that many of the RNN cells and functions are in the tf.contrib.rnn module). There was also a change to the ptb_iterator. A (slightly modified) copy of the old version which should work until I update this tutorial is uploaded [here](https://gist.github.com/spitis/2dd1720850154b25d2cec58d4b75c4a0).

### Recap of our model

In the last post, we built a very simple, no frills RNN that was quickly able to learn to solve the toy task we created for it.

Here is the formal statement of our model from last time:

$ S_t = \text{tanh}(W(X_t \ @ \ S_{t-1}) + b_s) $

$ P_t = \text{softmax}(US_t + b_p) $

where $ @ $ represents vector concatenation, $ X_t \in R^n $ is an input vector, $ W \in R^{d \times (n + d)}, \  b_s \in R^d, \ U \in R^{n \times d} $ is the size of the input and output vectors, and d is the size of the hidden state vector. At time step 0, $ S_{-1} $ (the initial state) is initialized as a vector of zeros.

### Task and data

This time around we will be building a character-level language model to generate character sequences, a la Andrej Karpathy’s [char-rnn](https://github.com/karpathy/char-rnn) (and see, e.g., a Tensorflow implementation by Sherjil Ozair [here](https://github.com/sherjilozair/char-rnn-tensorflow)).

Why do something that’s already been done? Well, this is a much harder task than the toy model from last time. This model needs to handle long sequences and learn long time dependencies. That makes a great task for learning about adding features to our RNN, and seeing how our changes affect the results as we go.

To start, let’s create our data generator. We’ll use the tiny-shakespeare corpus as our data, though we could use any plain text file. We’ll choose to use all of the characters in the text file as our vocabulary, treating lowercase and capital letters are separate characters. In practice, there may be some advantage to forcing the network to use similar representations for capital and lowercase letters by using the same one-hot representations for each, plus a binary flag to indicate whether or not the letter is a capital. Additionally, it is likely a good idea to restrict the vocabulary (i.e., the set of characters) used, by replacing uncommon characters with an UNK token (like a square: □).

{% highlight python %}
"""
Imports
"""
import numpy as np
import tensorflow as tf
%matplotlib inline
import matplotlib.pyplot as plt
import time
import os
import urllib.request
from tensorflow.models.rnn.ptb import reader
{% endhighlight %}

{% highlight python %}
"""
Load and process data, utility functions
"""

file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
file_name = 'tinyshakespeare.txt'
if not os.path.exists(file_name):
    urllib.request.urlretrieve(file_url, file_name)

with open(file_name,'r') as f:
    raw_data = f.read()
    print("Data length:", len(raw_data))

vocab = set(raw_data)
vocab_size = len(vocab)
idx_to_vocab = dict(enumerate(vocab))
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

data = [vocab_to_idx[c] for c in raw_data]
del raw_data

def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        yield reader.ptb_iterator(data, batch_size, num_steps)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def train_network(g, num_epochs, num_steps = 200, batch_size = 32, verbose = True, save=False):
    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1

                feed_dict={g['x']: X, g['y']: Y}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                      g['final_state'],
                                                      g['train_step']],
                                                             feed_dict)
                training_loss += training_loss_
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
            training_losses.append(training_loss/steps)

        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses
{% endhighlight %}

{% highlight python %}
Data length: 1115394
{% endhighlight %}

{% highlight python %}
def build_basic_rnn_graph_with_list(
    state_size = 100,
    num_classes = vocab_size,
    batch_size = 32,
    num_steps = 200,
    learning_rate = 1e-4):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    x_one_hot = tf.one_hot(x, num_classes)
    rnn_inputs = [tf.squeeze(i,squeeze_dims=[1]) for i in tf.split(1, num_steps, x_one_hot)]

    cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]

    y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, num_steps, y)]

    loss_weights = [tf.ones([batch_size]) for i in range(num_steps)]
    losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step
    )
{% endhighlight %}
