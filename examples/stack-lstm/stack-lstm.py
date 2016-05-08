'''
Part of this code is from theano tutorial:
  http://deeplearning.net/tutorial/lstm.html
'''
import copy
from collections import OrderedDict
import sys
import time
import pickle as pkl

import numpy as np
import theano
from theano import config
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from neuralcraft import layers, optimizers, utils

import imdb

def prepare_data_sp(x, y, maxlen=None):
  x, mask, y = imdb.prepare_data(x, y, maxlen)
  return (x.transpose(), mask.transpose(), y)
datasets = {'imdb': (imdb.load_data, prepare_data_sp)}

# Set the random number generators' seeds for consistency
SEED = 1234
np.random.seed(SEED)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]

def build_model(n_words, encoder, dim_proj, num_hidden, p_dropout, maxlen, decay_c, use_dropout=True, optimizer=optimizers.sgd):
  trng = RandomStreams(SEED)

  # by using shared variable, we can control whether to use noise without recompiling
  use_noise = theano.shared(utils.cast_floatX(0.))

  x = T.matrix('x', dtype='int64')
  xshape = (1, maxlen)
  mask = T.bmatrix('mask')
  y = T.vector('y', dtype='int64')
  lr = T.scalar()
  options = {'lr': lr}

  net = {}
  params = {}

  def init_emb(shape):
    num_in, num_out = shape
    randn = np.random.rand(num_in, num_out)
    return utils.cast_floatX(0.01 * randn)
  fc_winit = init_emb
  def ortho_weight(shape):
    ndim = shape[0]
    assert shape[0] == shape[1]
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return utils.cast_floatX(u)

  net['emb'] = layers.EmbeddingLayer((x, xshape), params, n_words, dim_proj, initializer=init_emb)
  if encoder == 'lstm':
    net['encoder'] = layers.LSTMLayer(net['emb'], 0., 0., params, num_hidden, mask, w_initializer=ortho_weight)
    '''
    shape = (dim_proj, num_hidden)
    net['encoder'] = layers.LSTMLayer(net['emb'], 0., 0., params, num_hidden, mask,
        w_xi=ortho_weight(shape), w_hi=ortho_weight(shape),
        w_xf=ortho_weight(shape), w_hf=ortho_weight(shape),
        w_xo=ortho_weight(shape), w_ho=ortho_weight(shape),
        w_xc=ortho_weight(shape), w_hc=ortho_weight(shape),
        )
        '''
  elif encoder == 'rnn':
    net['encoder'] = layers.RNNLayer(net['emb'], 0., params, num_hidden, mask, w_initializer=ortho_weight)

  net['mean_pool'] = (net['encoder'][0] * mask.dimshuffle((0, 1, 'x'))).sum(axis=1) / mask.sum(axis=1)[:, None].astype(theano.config.floatX) #mean pool. multiplied by mask to remove "EOS noise"
  encoder_shape = list(net['encoder'][1])
  net['mean_pool'] = (net['mean_pool'], [encoder_shape[0]] + encoder_shape[2:])
  if(use_dropout):
    net['dropout'] = layers.dropoutLayer(net['mean_pool'], use_noise, trng, p_dropout)

  pred, pred_shape = layers.FCLayer(net['dropout'], params, 2, activation=T.nnet.softmax, w_name='U', w_initializer=fc_winit)
  encoder = theano.function([x, mask], net['encoder'][0], name='encoder', allow_input_downcast=True)
  mean_pool = theano.function([x, mask], net['mean_pool'][0], name='mean_pool', allow_input_downcast=True)

  off = 1e-8
  #off = 0
  #cost = (T.nnet.categorical_crossentropy(pred, y)).mean() #correct
  n_samples = x.shape[0]
  cost = -T.log(pred[T.arange(n_samples), y] + off).mean()
  #cost_arr0 = T.nnet.categorical_crossentropy(pred, y)
  #cost_arr1 = -T.log(pred[T.arange(n_samples), y] + off)
  #n_samples = x.shape[0]
  if decay_c > 0:
    weight_decay = decay_c * (params['U'] ** 2).sum()
    cost += weight_decay


  f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob', allow_input_downcast=True)
  f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred', allow_input_downcast=True)
  opt = optimizer(cost, [x, mask, y], params, options=options)
  #opt = theano.function([x, mask, y], cost, allow_input_downcast=True)

  '''
  f_emb = theano.function([x], net['emb'][0], allow_input_downcast=True)
  f_encoder = theano.function([x, mask], net['encoder'][0], allow_input_downcast=True)
  f_dp = theano.function([x, mask], net['dropout'][0], allow_input_downcast=True)
  cost_arr0 = theano.function([x, mask, y], cost_arr0, allow_input_downcast=True)
  cost_arr1 = theano.function([x, mask, y], cost_arr1, allow_input_downcast=True)
  return f_pred_prob, f_pred, opt, params, use_noise, (f_emb, f_encoder, f_dp, cost_arr0, cost_arr1)
  '''
  return f_pred_prob, f_pred, opt, params, use_noise


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
  """
  Just compute the error
  f_pred: Theano fct computing the prediction
  prepare_data: usual prepare_data for that dataset.
  """
  valid_err = 0
  for _, valid_index in iterator:
    x, mask, y = prepare_data([data[0][t] for t in valid_index],
                              np.array(data[1])[valid_index],
                              maxlen=None)
    preds = f_pred(x, mask)
    targets = np.array(data[1])[valid_index]
    valid_err += (preds == targets).sum()
  print '-' * 80
  print preds
  print '-'*40
  print targets
  valid_err = 1. - utils.cast_floatX(valid_err) / len(data[0])

  return valid_err

def train_lstm(
    dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
    num_hidden=128,
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words=10000,  # Vocabulary size
    optimizer=optimizers.rmsprop,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='stack-lstm_model.pkl',  # The best model will be saved there
    validFreq=370,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='imdb',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    p_dropout=0.5,
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1 # If >0, we keep only this number of test example.
    ):
  load_data, prepare_data = get_dataset(dataset)
  train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
                                 maxlen=maxlen)
  #train = (train[0][:10], train[1][:10])

  if test_size > 0:
    # The test set is sorted by size, but we want to keep random
    # size example.  So we must select a random selection of the
    # examples.
    idx = np.arange(len(test[0]))
    np.random.shuffle(idx)
    idx = idx[:test_size]
    test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

  print('Building model')
  #f_pred_prob, f_pred, optimizer, params, use_noise, package = build_model(n_words, encoder, dim_proj, num_hidden, p_dropout, maxlen, decay_c, optimizer)
  f_pred_prob, f_pred, optimizer, params, use_noise = build_model(n_words, encoder, dim_proj, num_hidden, p_dropout, maxlen, decay_c, optimizer)
  use_noise.set_value(1.)

  print('Optimization')

  kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
  kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

  print("%d train examples" % len(train[0]))
  print("%d valid examples" % len(valid[0]))
  print("%d test examples" % len(test[0]))

  history_errs = []
  bad_counter = 0
  uidx = 0

  params_prev = None
  def compare_params(params, params_prev):
    m = True
    for p, pp in zip(params.values(), params_prev.values()):
      if p != pp:
        m = False
        break
    return m

  for eidx in range(max_epochs):
    print 'start epoch %d' % eidx
    kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
    count = 0
    cost_acc = 0

    for _, train_index in kf:
      use_noise.set_value(1.)

      # Select the random examples for this minibatch
      y = [train[1][t] for t in train_index]
      x = [train[0][t]for t in train_index]

      # Get the data in np.ndarray format
      # This swap the axis!
      # Return something of shape (minibatch maxlen, n samples)
      x, mask, y = prepare_data(x, y)
      count += 1

      '''
      print('-'*80)
      f_emb, f_enc, f_dp, cost_arr0, cost_arr1 = package
      print(f_emb(x).shape)
      print(f_enc(x, mask).shape)
      print(f_dp(x, mask).shape)
      print(f_pred_prob(x, mask).shape)
      print cost_arr0(x, mask, y)
      print cost_arr1(x, mask, y)
      prob = f_pred_prob(x, mask)
      #off = 1e-8
      #print("%.20f"%(- np.log(prob[range(prob.shape[0]), y] + off).mean()))
      '''

      cost = optimizer(x, mask, y, lrate)
      #print("%.20f"%cost)
      #cost = optimizer(x, mask, y)
      cost_acc += cost
      #print cost
      uidx += 1

      if uidx % dispFreq == 0:
        print "Epochs: %d, Updates: %d, Cost: %.9f" % (eidx, uidx, cost)
      if saveto and np.mod(uidx, saveFreq) == 0:
        print 'Saving to...'
        utils.save_params(saveto, params)

      if uidx % validFreq == 0:
        print params.values()[0].get_value()
        use_noise.set_value(0.)
        train_err = pred_error(f_pred, prepare_data, train, kf)
        valid_err = pred_error(f_pred, prepare_data, valid,
                               kf_valid)
        test_err = pred_error(f_pred, prepare_data, test, kf_test)

        history_errs.append([valid_err, test_err])

        print( ('Train ', train_err, 'Valid ', valid_err,
               'Test ', test_err) )

        if (len(history_errs) > patience and
            valid_err >= np.array(history_errs)[:-patience,
                                                   0].min()):
            bad_counter += 1
            if bad_counter > patience:
                print('Early Stop!')
                estop = True
                break

    print "Average cost for epoch %d: %f" % (eidx, cost_acc / count)


if __name__ == '__main__':
  # See function train for all possible parameter and there definition.
  train_lstm(
      #max_epochs=100,
      test_size=500,
      optimizer=optimizers.rmsprop,
      #encoder='rnn',
      #validFreq=20,
      lrate=0.1
  )
