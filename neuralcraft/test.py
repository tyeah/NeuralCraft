from layers import *
from optimizer import *
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

batch_size = 2
seq_size = 5
x_size = 3
hid_size = 4
output_size = 2
num_iter = 10
filter_size = 3
channel_size = 3
hid_channel_size = 6
dropout_p = 0.4

trng = RandomStreams(1234)

def FCTest():
  xt = T.matrix()
  yt = T.ivector()
  params = {}
  xout = FCLayer(xt, params, x_size, output_size)
  yhatt = T.nnet.softmax(xout)
  loss = T.mean(T.nnet.categorical_crossentropy(yhatt, yt))
  updates = sgd(loss, params)

  f = theano.function([xt, yt], loss, updates = updates, allow_input_downcast=True) 

  x = np.random.randn(batch_size, x_size)
  y = np.random.rand(batch_size) > 0.5
  for i in range(num_iter):
    print f(x, y)
    print [v.get_value() for v in params.values()]

def RNNTest():
  xt = T.tensor3()
  hidt = T.matrix()
  yt = T.imatrix()
  params = {}
  rnnout = RNNLayer(xt, hidt, params, x_size, hid_size)
  rnnout = rnnout.reshape((-1, rnnout.shape[-1]))
  # neet to reshape flatten first 2 dim of rnn output before inputing to FCLayer
  yhatt = FCLayer(rnnout, params, hid_size, output_size, activation=T.nnet.softmax)
  loss = T.mean(T.nnet.categorical_crossentropy(yhatt, yt.flatten()))
  updates = sgd(loss, params)

  f = theano.function([xt, hidt, yt], loss, updates = updates, allow_input_downcast=True) 

  x = np.random.randn(batch_size, seq_size, x_size)
  hid = np.random.randn(batch_size, hid_size)
  y = np.random.rand(batch_size, seq_size) > 0.5
  for i in range(num_iter):
    print f(x, hid, y)
    print [v.get_value() for v in params.values()]

def LSTMTest():
  xt = T.tensor3()
  hidt = T.matrix()
  cellt = T.matrix()
  yt = T.imatrix()
  params = {}
  #rnnout = LSTMLayer(xt, hidt, cellt, params, x_size, hid_size)
  rnnout = LSTMLayer(xt, 0, np.random.randn(hid_size), params, x_size, hid_size)
  rnnout = rnnout.reshape((-1, rnnout.shape[-1]))
  # neet to reshape flatten first 2 dim of rnn output before inputing to FCLayer
  yhatt = FCLayer(rnnout, params, hid_size, output_size, activation=T.nnet.softmax)
  loss = T.mean(T.nnet.categorical_crossentropy(yhatt, yt.flatten()))
  updates = sgd(loss, params)

  #f = theano.function([xt, hidt, cellt, yt], loss, updates = updates, allow_input_downcast=True) 
  f = theano.function([xt, yt], loss, updates = updates, allow_input_downcast=True) 

  x = np.random.randn(batch_size, seq_size, x_size)
  hid = np.random.randn(batch_size, hid_size)
  cell = np.random.randn(batch_size, hid_size)
  y = np.random.rand(batch_size, seq_size) > 0.5
  for i in range(num_iter):
    #print f(x, hid, cell, y)
    print f(x, y)
    print [v.get_value() for v in params.values()]

def Conv2DTest():
  x = np.random.randn(batch_size, channel_size, x_size, x_size)
  y = np.random.rand(batch_size) > 0.5

  xt = T.tensor4()
  yt = T.ivector()
  params = {}
  cnnout = Conv2DLayer(xt, params, channel_size, hid_channel_size, filter_size)
  f = theano.function([xt], cnnout, allow_input_downcast=True) 
  print f(x).shape


  yhatt = FCLayer(cnnout, params, x_size * x_size * hid_channel_size, output_size, activation=T.nnet.softmax)
  loss = T.mean(T.nnet.categorical_crossentropy(yhatt, yt.flatten()))
  updates = sgd(loss, params)

  #f = theano.function([xt, hidt, cellt, yt], loss, updates = updates, allow_input_downcast=True) 
  f = theano.function([xt, yt], loss, updates = updates, allow_input_downcast=True) 

  for i in range(num_iter):
    #print f(x, hid, cell, y)
    print f(x, y)
    print [v.get_value() for v in params.values()]

def dropoutTest():
  xt = T.tensor3()
  use_noise = T.bscalar()
  yt = dropoutLayer(xt, use_noise, trng, dropout_p)
  f = theano.function([xt, use_noise], yt, allow_input_downcast=True) 

  x = np.random.randn(batch_size, seq_size, x_size)
  print f(x, True)

  use_noise = 1
  yt = dropoutLayer(xt, use_noise, trng, dropout_p)
  f = theano.function([xt], yt, allow_input_downcast=True) 

  x = np.random.randn(batch_size, seq_size, x_size)
  print f(x)

def poolingTest():
  xt = T.tensor3()
  yt = poolingLayer(xt, ds_h=2)
  f = theano.function([xt], yt, allow_input_downcast=True) 

  x = np.random.randn(batch_size, 10, 10)
  print f(x)

if __name__ == '__main__':
  #FCTest()
  #RNNTest()
  #LSTMTest()
  #Conv2DTest()
  #dropoutTest()
  poolingTest()
