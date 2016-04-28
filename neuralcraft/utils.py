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
