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

### Using tf.scan and dynamic_rnn to speed things up

Recall from [last post](https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html) that we represented each duplicate tensor of our RNN (e.g., the rnn inputs, rnn outputs, the predictions and the loss) as a list of tensors:

<p align="center">
![](https://r2rt.com/static/images/BasicRNNLabeled.png)
</p>

This worked quite well for our toy task, because our longest dependency was 7 steps back and we never really needed to backpropagate errors more than 10 steps. Even with a word-level RNN, using lists will probably be sufficient. See, e.g., my post on [Styles of Truncated Backpropagation](http://r2rt.com/styles-of-truncated-backpropagation.html), where I build a 40-step graph with no problems. But for a character-level model, 40 characters isn’t a whole lot. We might want to capture much longer dependencies. So let’s see what happens when we build a graph that is 200 time steps wide:

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

{% highlight python %}
t = time.time()
build_basic_rnn_graph_with_list()
print("It took", time.time() - t, "seconds to build the graph.")
{% endhighlight %}

{% highlight python %}
It took 5.626644849777222 seconds to build the graph.
{% endhighlight %}

It took over 5 seconds to build the graph of the most basic RNN model! This could bad… what happens when we move up to a 3-layer LSTM?

Below, we switch out the RNN cell for a Multi-layer LSTM cell. We’ll go over the details of how to do this in the next section.

{% highlight python %}
def build_multilayer_lstm_graph_with_list(
    state_size = 100,
    num_classes = vocab_size,
    batch_size = 32,
    num_steps = 200,
    num_layers = 3,
    learning_rate = 1e-4):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])
    rnn_inputs = [tf.squeeze(i) for i in tf.split(1,
                                num_steps, tf.nn.embedding_lookup(embeddings, x))]

    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
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

{% highlight python %}
t = time.time()
build_multilayer_lstm_graph_with_list()
print("It took", time.time() - t, "seconds to build the graph.")
{% endhighlight %}

{% highlight python %}
It took 25.640846967697144 seconds to build the graph.
{% endhighlight %}

Yikes, almost 30 seconds.

Now this isn’t that big of an issue for training, because we only need to build the graph once. It could be a big issue, however, if we need to build the graph multiple times at test time.

To get around this long compile time, Tensorflow allows us to create the graph at runtime. Here is a quick demonstration of the difference, using Tensorflow’s dynamic_rnn function:

{% highlight python %}
def build_multilayer_lstm_graph_with_dynamic_rnn(
    state_size = 100,
    num_classes = vocab_size,
    batch_size = 32,
    num_steps = 200,
    num_layers = 3,
    learning_rate = 1e-4):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

    # Note that our inputs are no longer a list, but a tensor of dims batch_size x num_steps x state_size
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    #reshape rnn_outputs and y so we can get the logits in a single matmul
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
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

{% highlight python %}
t = time.time()
build_multilayer_lstm_graph_with_dynamic_rnn()
print("It took", time.time() - t, "seconds to build the graph.")
{% endhighlight %}

{% highlight python %}
It took 0.5314393043518066 seconds to build the graph.
{% endhighlight %}

Much better. One would think that pushing the graph construction to execution time would cause execution of the graph to go slower, but in this case, using dynamic_rnn actually speeds things up:

{% highlight python %}
g = build_multilayer_lstm_graph_with_list()
t = time.time()
train_network(g, 3)
print("It took", time.time() - t, "seconds to train for 3 epochs.")
{% endhighlight %}

{% highlight python %}
Average training loss for Epoch 0 : 3.53323210245
Average training loss for Epoch 1 : 3.31435756163
Average training loss for Epoch 2 : 3.21755325109
It took 117.78161263465881 seconds to train for 3 epochs.
{% endhighlight %}

{% highlight python %}
g = build_multilayer_lstm_graph_with_dynamic_rnn()
t = time.time()
train_network(g, 3)
print("It took", time.time() - t, "seconds to train for 3 epochs.")
{% endhighlight %}

{% highlight python %}
Average training loss for Epoch 0 : 3.55792756053
Average training loss for Epoch 1 : 3.3225021006
Average training loss for Epoch 2 : 3.28286816745
It took 96.69413661956787 seconds to train for 3 epochs.
{% endhighlight %}

It’s not a breeze to work through and understand the dynamic_rnn code (which lives [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py)), but we can obtain a similar result ourselves by using tf.scan (dynamic_rnn does not use scan). Scan runs just a tad slower than Tensorflow’s optimized code, but is easier to understand and write yourself.

Scan is a higher-order function that you might be familiar with if you’ve done any programming in OCaml, Haskell or the like. In general, it takes a function $ (f: (x_t, y_{t-1}) \mapsto y_t) $, a sequence $ ([x_0, x_1 \dots x_n]) $ and an initial value $ (y_{-1}) $ and returns a sequence $ ([y_0, y_1 \dots y_n]) $ according to the rule: $ y_t = f(x_t, y_{t-1}) $. In Tensorflow, scan treats the first dimension of a Tensor as the sequence. Thus, if fed a Tensor of shape [n, m, o] as the sequence, scan would unpack it into a sequence of n-tensors, each with shape [m, o]. You can learn more about Tensorflow’s scan [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functional_ops.md#tfscanfn-elems-initializernone-parallel_iterations10-back_proptrue-swap_memoryfalse-namenone-scan).

Below, I use scan with an LSTM so as to compare to the dynamic_rnn using Tensorflow above. Because LSTMs store their state in a 2-tuple, and we’re using a 3-layer network, the scan function produces, as `final_states below`, a 3-tuple (one for each layer) of 2-tuples (one for each LSTM state), each of shape [num_steps, batch_size, state_size]. We need only the last state, which is why we unpack, slice and repack `final_states` to get `final_state` below.

Another thing to note is that scan produces rnn_outputs with shape [num_steps, batch_size, state_size], whereas the dynamic_rnn produces rnn_outputs with shape [batch_size, num_steps, state_size] (the first two dimensions are switched). Dynamic_rnn has the flexibility to switch this behavior, using the “time_major” argument. Tf.scan does not have this flexibility, which is why we transpose `rnn_inputs` and `y` in the code below.

{% highlight python %}
def build_multilayer_lstm_graph_with_scan(
    state_size = 100,
    num_classes = vocab_size,
    batch_size = 32,
    num_steps = 200,
    num_layers = 3,
    learning_rate = 1e-4):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_states = \
        tf.scan(lambda a, x: cell(x, a[1]),
                tf.transpose(rnn_inputs, [1,0,2]),
                initializer=(tf.zeros([batch_size, state_size]), init_state))

    # there may be a better way to do this:
    final_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
                  tf.squeeze(tf.slice(c, [num_steps-1,0,0], [1, batch_size, state_size])),
                  tf.squeeze(tf.slice(h, [num_steps-1,0,0], [1, batch_size, state_size])))
                       for c, h in final_states])

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(tf.transpose(y,[1,0]), [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
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

{% highlight python %}
t = time.time()
g = build_multilayer_lstm_graph_with_scan()
print("It took", time.time() - t, "seconds to build the graph.")
t = time.time()
train_network(g, 3)
print("It took", time.time() - t, "seconds to train for 3 epochs.")
{% endhighlight %}

{% highlight python %}
It took 0.6475389003753662 seconds to build the graph.
Average training loss for Epoch 0 : 3.55362293501
Average training loss for Epoch 1 : 3.32045680079
Average training loss for Epoch 2 : 3.27433713688
It took 101.60246014595032 seconds to train for 3 epochs.
{% endhighlight %}

Using scan was only marginally slower than using dynamic_rnn, and gives us the flexibility and understanding to tweak the code if we ever need to (e.g., if for some reason we wanted to create a skip connection from the state at timestep t-2 to timestep t, it would be easy to do with scan).
