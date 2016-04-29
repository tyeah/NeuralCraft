import theano
from theano import tensor as T
from theano.tensor import nnet
import numpy as np
import pickle as pkl


def save_params(filepath, params):
  pkl.dump(params, open(filepath, 'w'));


def load_params(filepath):
  return pkl.load(open(filepath, 'r'));


def cross_entropy(yhat, y):
  last_dim_len = y.shape[-1]
  if y.ndim == yhat.ndim:
    #y is one-hot
    yhat = T.reshape(-1, last_dim_len)
    y = T.reshape(-1, last_dim_len)
  elif y.ndim == yhat.ndim + 1:
    yhat = T.reshape(-1, last_dim_len)
    y = T.flatten(y)
  return T.mean(nnet.categorical_crossentropy(yhatt, yt))


def reshape(shape_prev, shape_after):
  if np.prod(shape_prev) == np.prod(shape_after):
    return shape_after
  assert np.prod(shape_after) < 0, "shape product changed: %s vs %s" % ((shape_prev,), (shape_after,))
  id = np.where(np.array(shape_after) < 0)[0]
  assert len(id) == 1, "more than one negative dim"
  id = id[0]
  dim_id = int(np.prod(shape_prev) / (np.prod(shape_after[:id]) * np.prod(shape_after[(id+1):])))
  shape_out = list(shape_after)
  shape_out[id] = dim_id
  assert np.prod(shape_prev) == np.prod(shape_out), "inferred shape product changed"
  return tuple(shape_out)

def cast_floatX(x):
  return np.asarray(x, dtype=theano.config.floatX)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
  '''
  used to sample minibatches from data with ndarray type
  '''
  assert len(inputs) == len(targets)
  if shuffle:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
  for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
    if shuffle:
        excerpt = indices[start_idx:start_idx + batchsize]
    else:
        excerpt = slice(start_idx, start_idx + batchsize)
    yield inputs[excerpt], targets[excerpt]

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration and generate indices for a minibatch.
    return:
      a list of (i, indices_in_minibatch_i) tuples
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
