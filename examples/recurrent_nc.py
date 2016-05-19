#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Recurrent network example.  Trains a bidirectional vanilla RNN to output the
sum of two numbers in a sequence of random numbers sampled uniformly from
[0, 1] based on a separate marker sequence.
'''

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne

from neuralcraft import layers
from neuralcraft import optimizers
from neuralcraft import utils
from neuralcraft import init
from neuralcraft.utils import cast_floatX


# Min/max sequence length
MIN_LENGTH = 50
MAX_LENGTH = 55
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# Number of training sequences in each batch
N_BATCH = 100
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 10

SEED = 1234
np.random.seed(SEED)


def gen_data(min_length=MIN_LENGTH, max_length=MAX_LENGTH, n_batch=N_BATCH):
    '''
    Generate a batch of sequences for the "add" task, e.g. the target for the
    following

    ``| 0.5 | 0.7 | 0.3 | 0.1 | 0.2 | ... | 0.5 | 0.9 | ... | 0.8 | 0.2 |
      |  0  |  0  |  1  |  0  |  0  |     |  0  |  1  |     |  0  |  0  |``

    would be 0.3 + .9 = 1.2.  This task was proposed in [1]_ and explored in
    e.g. [2]_.

    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    max_length : int
        Maximum sequence length.
    n_batch : int
        Number of samples in the batch.

    Returns
    -------
    X : np.ndarray
        Input to the network, of shape (n_batch, max_length, 2), where the last
        dimension corresponds to the two sequences shown above.
    y : np.ndarray
        Correct output for each sample, shape (n_batch,).
    mask : np.ndarray
        A binary matrix of shape (n_batch, max_length) where ``mask[i, j] = 1``
        when ``j <= (length of sequence i)`` and ``mask[i, j] = 0`` when ``j >
        (length of sequence i)``.

    References
    ----------
    .. [1] Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory."
    Neural computation 9.8 (1997): 1735-1780.

    .. [2] Sutskever, Ilya, et al. "On the importance of initialization and
    momentum in deep learning." Proceedings of the 30th international
    conference on machine learning (ICML-13). 2013.
    '''
    # Generate X - we'll fill the last dimension later
    X = np.concatenate([np.random.uniform(size=(n_batch, max_length, 1)),
                        np.zeros((n_batch, max_length, 1))],
                       axis=-1)
    mask = np.zeros((n_batch, max_length))
    y = np.zeros((n_batch,))
    # Compute masks and correct values
    for n in range(n_batch):
        # Randomly choose the sequence length
        length = np.random.randint(min_length, max_length)
        # Make the mask for this sample 1 within the range of length
        mask[n, :length] = 1
        # Zero out X after the end of the sequence
        X[n, length:, 0] = 0
        # Set the second dimension to 1 at the indices to add
        X[n, np.random.randint(length/10), 1] = 1
        X[n, np.random.randint(length/2, length), 1] = 1
        # Multiply and sum the dimensions of X to get the target value
        y[n] = np.sum(X[n, :, 0]*X[n, :, 1])
    # Center the inputs and outputs
    X -= X.reshape(-1, 2).mean(axis=0)
    y -= y.mean()
    return (X.astype(theano.config.floatX), y.astype(theano.config.floatX),
            mask.astype(theano.config.floatX))


def main(num_epochs=NUM_EPOCHS):
    # set weights
    w_xh = init.HeUniform().sample((2, N_HIDDEN))
    w_hh = init.HeUniform().sample((N_HIDDEN, N_HIDDEN))
    b_rnn = np.zeros(N_HIDDEN)
    w_fc = init.HeUniform().sample((N_HIDDEN, 1))
    print(w_fc[:10])
    b_fc = init.Const(0.).sample(1)
    print("Building network ...")
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    #l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, 2))
    params = {}
    l_in = (T.tensor3(), (N_BATCH, MAX_LENGTH, 2))

    # The network also needs a way to provide a mask for each sequence.  We'll
    # use a separate input layer for that.  Since the mask only determines
    # which indices are part of the sequence for each batch entry, they are
    # supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
    #l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))
    l_mask = T.imatrix()

    # We're using a bidirectional network, which means we will combine two
    # RecurrentLayers, one with the backwards=True keyword argument.
    # Setting a value for grad_clipping will clip the gradients in the layer
    # Setting only_return_final=True makes the layers only return their output
    # for the final time step, which is all we need for this task
    '''
    l_forward = layers.RNNLayer(
        l_in, 0., params, N_HIDDEN, mask=l_mask,
        w_xh=w_xh, w_hh=w_hh, b=b_rnn,
        activation=T.tanh, only_return_final=True)
    '''
    #LSTM Version
    l_forward = layers.LSTMLayer(
        l_in, 0., 0., params, N_HIDDEN, mask=l_mask,
        activation=T.tanh, only_return_final=True)
    
    # Our output layer is a simple dense connection, with 1 output unit
    l_out = layers.FCLayer(
        l_forward, params, num_out=1, activation=T.tanh,
        w=w_fc, b=b_fc)

    target_values = T.vector('target_output')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = l_out[0]
    # The network output will have shape (n_batch, 1); let's flatten to get a
    # 1-dimensional vector of predicted values
    predicted_values = network_output.flatten()
    # Our cost will be mean-squared error
    cost = T.mean((predicted_values - target_values)**2)
    # Retrieve all parameters from the network
    # Compute SGD updates for training
    print("Computing updates ...")
    train = optimizers.adagrad(cost, (l_in[0], target_values, l_mask), params, {'lr': LEARNING_RATE})
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    compute_cost = theano.function(
        [l_in[0], target_values, l_mask], cost, allow_input_downcast=True)

    # We'll use this "validation set" to periodically check progress
    X_val, y_val, mask_val = gen_data()

    print("Training ...")
    try:
        cost_val = compute_cost(X_val, y_val, mask_val)
        print("validation cost before training= {}".format(cost_val))
        for epoch in range(num_epochs):
            for _ in range(EPOCH_SIZE):
                X, y, m = gen_data()
                train(X, y, m)
            cost_val = compute_cost(X_val, y_val, mask_val)
            print("Epoch {} validation cost = {}".format(epoch, cost_val))
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
