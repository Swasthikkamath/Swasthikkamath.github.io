---
layout: post
title:  "Recurrent Neural Networks in Tensorflow II"
date:   2022-01-04 11:28:34 +0530
categories: jekyll tensorflow neuralNetworks
---

This is the second in a series of posts about recurrent neural networks in Tensorflow. The first post lives [here](https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html). In this post, we will build upon our vanilla RNN by learning how to use Tensorflow’s scan and dynamic_rnn models, upgrading the RNN cell and stacking multiple RNNs, and adding dropout and layer normalization. We will then use our upgraded RNN to generate some text, character by character.

**Note 3/14/2017:** This tutorial is quite a bit deprecated by changes to the TF api. Leaving it up since it may still be useful, and most changes to the API are cosmetic (biggest change is that many of the RNN cells and functions are in the tf.contrib.rnn module). There was also a change to the ptb_iterator. A (slightly modified) copy of the old version which should work until I update this tutorial is uploaded [here].

### Recap of our model

In the last post, we built a very simple, no frills RNN that was quickly able to learn to solve the toy task we created for it.

Here is the formal statement of our model from last time:

$S_t = \text{tanh}(W(X_t \ @ \ S_{t-1}) + b_s)$
$P_t = \text{softmax}(US_t + b_p)$

where $@$ represents vector concatenation, $X_t \in R^n$ is an input vector, $W \in R^{d \times (n + d)}, \  b_s \in R^d, \ U \in R^{n \times d}$ is the size of the input and output vectors, and d is the size of the hidden state vector. At time step 0, $S_{-1}$ (the initial state) is initialized as a vector of zeros.

### Task and data

This time around we will be building a character-level language model to generate character sequences, a la Andrej Karpathy’s [char-rnn](https://github.com/karpathy/char-rnn) (and see, e.g., a Tensorflow implementation by Sherjil Ozair [here](https://github.com/sherjilozair/char-rnn-tensorflow)).

Why do something that’s already been done? Well, this is a much harder task than the toy model from last time. This model needs to handle long sequences and learn long time dependencies. That makes a great task for learning about adding features to our RNN, and seeing how our changes affect the results as we go.

To start, let’s create our data generator. We’ll use the tiny-shakespeare corpus as our data, though we could use any plain text file. We’ll choose to use all of the characters in the text file as our vocabulary, treating lowercase and capital letters are separate characters. In practice, there may be some advantage to forcing the network to use similar representations for capital and lowercase letters by using the same one-hot representations for each, plus a binary flag to indicate whether or not the letter is a capital. Additionally, it is likely a good idea to restrict the vocabulary (i.e., the set of characters) used, by replacing uncommon characters with an UNK token (like a square: □).
