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

'''
def reshape(shape_prev, shape_after):
  batch_size_prev, shape_prev = shape_prev[0], shape_prev[1:]
  batch_size_after, shape_after = shape_after[0], shape_after[1:]
  assert batch_size_prev == batch_size_after or \
      (batch_size_prev == 'x' and batch_size_after == 'x'), 'batch size must be the same'
  if np.prod(shape_prev) == np.prod(shape_after):
    return tuple([batch_size_prev] + list(shape_after))
  #assert np.prod(shape_after) < 0, "shape product changed: %s vs %s" % ((shape_prev,), (shape_after,))
  id = np.where(np.array(shape_after) < 0)[0]
  assert (len(id) == 1 and batch_size_prev != 'x') or (len(id) == 0 and batch_size_prev == 'x'),\
      "more than one uncertain dim"
  if batch_size_prev == 'x':
    assert np.prod(shape_prev) % np.prod(shape_after) == 0, \
        "shape product does not devide the original shape product" 
    return tuple([batch_size_prev] + list(shape_after))
  else:
    id = id[0]
    whole_prod = np.prod(shape_prev)
    res_prod = np.prod(shape_after[:id]) * np.prod(shape_after[(id+1):])
    assert whole_prod % res_prod == 0, "shape product does not devide the original shape product" 
    dim_id = int(whole_prod / res_prod)
    shape_out = list(shape_after)
    shape_out[id] = dim_id
    assert np.prod(shape_prev) == np.prod(shape_out), "inferred shape product changed"
    return tuple([batch_size_prev] + shape_out)
  '''

def reshape(shape_prev, shape_after):
  batch_size_prev, shape_prev = shape_prev[0], shape_prev[1:]
  batch_size_after, shape_after = shape_after[0], shape_after[1:]
  shape_prev = list(shape_prev)
  shape_after = list(shape_after)
  if batch_size_prev == 'x':
    assert batch_size_after == 'x', \
        "when input batch size is uncertain ('x'), output batch size should also be 'x')"
    assert np.prod(shape_after) > 0, "only one axis can be uncertain"
    assert np.prod(shape_prev) % np.prod(shape_after) == 0, \
        "shape product does not devide the original shape product" 
    return tuple([batch_size_prev] + shape_after)
  if np.prod([batch_size_prev] + shape_prev) == np.prod([batch_size_after] + shape_after):
    return tuple([batch_size_prev] + list(shape_after))
  assert np.prod(shape_after) < 0, "shape product changed: %s vs %s" % ((shape_prev,), (shape_after,))
  id = np.where(np.array(shape_after) < 0)[0]
  assert len(id) == 1, "more than one uncertain dim"
  id = id[0]
  whole_prod = np.prod(shape_prev)
  res_prod = np.prod(shape_after[:id]) * np.prod(shape_after[(id+1):])
  assert whole_prod % res_prod == 0, "shape product does not devide the original shape product" 
  dim_id = int(whole_prod / res_prod)
  shape_out = list(shape_after)
  shape_out[id] = dim_id
  assert np.prod(shape_prev) == np.prod(shape_out), "inferred shape product changed"
  return tuple([batch_size_prev] + shape_out)

def layer_reshape(layer, shape_after):
  shape_after = reshape(layer[1], shape_after)
  if shape_after[0] == 'x':
    output = layer[0].reshape([-1] + list(shape_after)[1:])
  else:
    output = layer[0].reshape(shape_after)
  return (output, shape_after)

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
