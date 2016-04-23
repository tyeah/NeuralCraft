from layers import *
from optimizer import *
import numpy as np

batch_size = 2
seq_size = 5
x_size = 3
hid_size = 4
output_size = 2
num_iter = 10

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


if __name__ == '__main__':
  #FCTest()
  #RNNTest()
  LSTMTest()
