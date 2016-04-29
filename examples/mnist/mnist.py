'''
Part of this code is from lasagne:
  https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
'''
import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
from theano.tensor import nnet
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from neuralcraft import layers, utils, optimizers

SEED = 123
np.random.seed(SEED)

def load_dataset():
  # We first define a download function, supporting both Python 2 and 3.
  if sys.version_info[0] == 2:
    from urllib import urlretrieve
  else:
    from urllib.request import urlretrieve

  def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)

  # We then define functions for loading MNIST images and labels.
  # For convenience, they also download the requested files if needed.
  import gzip

  def load_mnist_images(filename):
    if not os.path.exists(filename):
      download(filename)
    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    data = data.reshape(-1, 1, 28, 28)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / np.float32(256)

  def load_mnist_labels(filename):
    if not os.path.exists(filename):
      download(filename)
    # Read the labels in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data

  # We can now download and read the training and test set images and labels.
  X_train = load_mnist_images('train-images-idx3-ubyte.gz')
  y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
  X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
  y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

  # We reserve the last 10000 training examples for validation.
  X_train, X_val = X_train[:-10000], X_train[-10000:]
  y_train, y_val = y_train[:-10000], y_train[-10000:]

  # We just return all the arrays in order, as expected in main().
  # (It doesn't matter how we do this as long as we can read them again.)
  return X_train, y_train, X_val, y_val, X_test, y_test

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
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

def build_mlp(input_shape=(1, 1, 28, 28), optimizer=optimizers.sgd):
  trng = RandomStreams(SEED)

  x = T.tensor4()
  y = T.ivector()
  lr = T.scalar()
  options = {'lr': lr}

  x_in = (x, input_shape)
  params = {}
  net = {}
  net['hid1'] = layers.FCLayer(x_in, params, 500)
  net['hid2'] = layers.FCLayer(net['hid1'], params, 500)
  pred, _ = layers.FCLayer(net['hid2'], params, 500, nnet.softmax)
  cost = T.nnet.categorical_crossentropy(pred, y).mean()

  f_pred_prob = theano.function([x], pred, name='f_pred_prob', allow_input_downcast=True)
  f_pred = theano.function([x], pred.argmax(axis=1), name='f_pred', allow_input_downcast=True)
  opt = optimizer(cost, [x, y], params, options)
  return f_pred_prob, f_pred, opt, params

def pred_acc(f_pred, x, y):
  preds = f_pred(x)
  acc = (preds == y).mean()
  return acc

def sample_data(X, y, batch_size):
  l = X.shape[0]
  idx = np.random.choice(range(l), batch_size)
  return (X[idx], y[idx])

def train(lrate=0.01, num_epochs=100, dispFreq=5, optimizer=optimizers.rmsprop):
  X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
  f_pred_prob, f_pred, optimizer, params = build_mlp(input_shape=(1, 1, 28, 28), optimizer=optimizer)
  for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    cost = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
      inputs, targets = batch
      cost += optimizer(inputs, targets, lrate)
      train_batches += 1
    print " Average training cost of epoch %d: %f" % (epoch, (cost / train_batches))
    if epoch % dispFreq == 0:
      X_train_test, y_train_test = sample_data(X_train, y_train, 200)
      X_val_test, y_val_test = sample_data(X_val, y_val, 200)
      train_acc = pred_acc(f_pred, X_train_test, y_train_test)
      val_acc = pred_acc(f_pred, X_val_test, y_val_test)
      print "training accuracy: %f, validation accuracy: %f" % (train_acc, val_acc)
    

if __name__ == '__main__':
  train(optimizer=optimizers.adam, dispFreq=5)
