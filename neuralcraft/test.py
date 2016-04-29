from layers import *
from optimizers import *
from utils import *
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

batch_size = 2
seq_size = 5
x_size = 3
hid_size = 4
output_size = 2
num_iter = 2
filter_size = 3
channel_size = 3
hid_channel_size = 6
dropout_p = 0.4
vocab_size = 8
embedding_size = 9

trng = RandomStreams(1234)

def optimizerTest(optimizer=sgd):
  x = np.random.randn(batch_size, x_size)
  y = np.random.rand(batch_size) > 0.5

  xt = T.matrix()
  xt = (xt, x.shape)
  yt = T.ivector()
  lr = T.scalar()
  params = {}
  xout = FCLayer(xt, params, output_size)
  yhatt = T.nnet.softmax(xout[0])
  loss = T.mean(T.nnet.categorical_crossentropy(yhatt, yt))

  options = {'lr': lr}
  f_update = optimizer(loss, [xt[0], yt], params, options=options)
  for i in range(num_iter):
    print f_update(x, y, 0.01)
    print [v.get_value() for v in params.values()]

def FCTest():
  x = np.random.randn(batch_size, x_size)
  xt = T.matrix()
  xt = (xt, x.shape)
  params = {}
  xout = FCLayer(xt, params, output_size)
  f = theano.function([xt[0]], xout, updates = updates, allow_input_downcast=True) 
  print f(x)

def RNNTest():
  x = np.random.randn(batch_size, seq_size, x_size)
  mask = np.random.rand(batch_size, seq_size) > 0.5
  hid = np.random.randn(batch_size, hid_size)
  y = np.random.rand(batch_size, seq_size) > 0.5

  xt = T.tensor3()
  xt = (xt, x.shape)
  maskt = T.bmatrix()
  hidt = T.matrix()
  yt = T.imatrix()
  params = {}

  #without mask
  rnnout, rnnout_shape = RNNLayer(xt, hidt, params, hid_size, only_return_final=True)
  f = theano.function([xt[0], hidt], rnnout, allow_input_downcast=True)
  print 'without mask:'
  print f(x, hid)
  #with mask
  params = {}
  rnnout, rnnout_shape = RNNLayer(xt, hidt, params, hid_size, maskt)
  f = theano.function([xt[0], hidt, maskt], rnnout, allow_input_downcast=True)
  print 'with mask:'
  print mask
  print f(x, hid, mask)

  rnnout = rnnout.reshape((-1, rnnout.shape[-1]))
  rnnout_shape = (np.prod(rnnout_shape[-1:]), rnnout_shape[-1])
  # neet to reshape flatten first 2 dim of rnn output before inputing to FCLayer
  yhatt, _ = FCLayer((rnnout, rnnout_shape), params, output_size, activation=T.nnet.softmax)
  loss = T.mean(T.nnet.categorical_crossentropy(yhatt, yt.flatten()))
  updates = sgd(loss, params)

  f = theano.function([xt[0], hidt, maskt, yt], loss, updates = updates, allow_input_downcast=True) 

  for i in range(num_iter):
    print f(x, hid, mask, y)
    #print [v.get_value() for v in params.values()]

def LSTMTest():
  x = np.random.randn(batch_size, seq_size, x_size)
  mask = np.random.rand(batch_size, seq_size) > 0.5
  hid = np.random.randn(batch_size, hid_size)
  cell = np.random.randn(batch_size, hid_size)
  y = np.random.rand(batch_size, seq_size) > 0.5

  xt = T.tensor3()
  xt = (xt, x.shape)
  maskt = T.bmatrix()
  hidt = T.matrix()
  cellt = T.matrix()
  yt = T.imatrix()
  params = {}

  rnnout, rnnout_shape = LSTMLayer(xt, 0, np.random.randn(hid_size), params, hid_size, maskt, only_return_final=True)
  f = theano.function([xt[0], maskt], rnnout, allow_input_downcast=True)
  print 'with mask:'
  print mask
  print f(x, mask)

  params = {}
  rnnout, rnnout_shape = LSTMLayer(xt, 0, np.random.randn(hid_size), params, hid_size)
  f = theano.function([xt[0]], rnnout, allow_input_downcast=True)
  print 'without mask:'
  print f(x)

  rnnout = rnnout.reshape((-1, rnnout.shape[-1]))
  rnnout_shape = (np.prod(rnnout_shape[-1:]), rnnout_shape[-1])
  # neet to reshape flatten first 2 dim of rnn output before inputing to FCLayer
  yhatt, _ = FCLayer((rnnout, rnnout_shape), params, output_size, activation=T.nnet.softmax)
  loss = T.mean(T.nnet.categorical_crossentropy(yhatt, yt.flatten()))
  updates = sgd(loss, params)

  #f = theano.function([xt, hidt, cellt, yt], loss, updates = updates, allow_input_downcast=True) 
  f = theano.function([xt[0], yt], loss, updates = updates, allow_input_downcast=True) 

  print 'f and params:'
  for i in range(num_iter):
    print f(x, y)
    print [v.get_value() for v in params.values()]

def Conv2DTest():
  x = np.random.randn(batch_size, channel_size, x_size, x_size)
  y = np.random.rand(batch_size) > 0.5

  xt = T.tensor4()
  xt = (xt, x.shape)
  yt = T.ivector()
  params = {}
  cnnout = Conv2DLayer(xt, params, hid_channel_size, filter_size)
  f = theano.function([xt[0]], cnnout[0], allow_input_downcast=True) 
  print f(x).shape, cnnout[1]


  yhatt, _ = FCLayer(cnnout, params, output_size, activation=T.nnet.softmax)
  loss = T.mean(T.nnet.categorical_crossentropy(yhatt, yt.flatten()))
  updates = sgd(loss, params)

  #f = theano.function([xt, hidt, cellt, yt], loss, updates = updates, allow_input_downcast=True) 
  f = theano.function([xt[0], yt], loss, updates = updates, allow_input_downcast=True) 

  for i in range(num_iter):
    #print f(x, hid, cell, y)
    print f(x, y)
    print [v.get_value() for v in params.values()]

def dropoutTest():
  x = np.random.randn(batch_size, seq_size, x_size)

  xt = T.tensor3()
  xt = (xt, x.shape)
  use_noise = T.bscalar()
  yt, _ = dropoutLayer(xt, use_noise, trng, dropout_p)
  f = theano.function([xt[0], use_noise], yt, allow_input_downcast=True) 

  x = np.random.randn(batch_size, seq_size, x_size)
  print f(x, True)

  use_noise = 1
  yt, _ = dropoutLayer(xt, use_noise, trng, dropout_p)
  f = theano.function([xt[0]], yt, allow_input_downcast=True) 

  print f(x)

def poolingTest():
  x = np.random.randn(batch_size, 10, 10)

  xt = T.tensor3()
  xt = (xt, x.shape)
  yt, pool_shape = poolingLayer(xt, ds_h=2)
  f = theano.function([xt[0]], yt, allow_input_downcast=True) 

  print pool_shape
  print f(x)

def EmbeddingTest():
  x = np.random.choice(range(vocab_size), [batch_size, x_size])

  xt = T.imatrix()
  xt = (xt, x.shape)
  params = {}
  yt, y_shape = EmbeddingLayer(xt, params, vocab_size, embedding_size)
  print y_shape

  f = theano.function([xt[0]], yt, allow_input_downcast=True)

  print f(x)

if __name__ == '__main__':
  #FCTest()
  #RNNTest()
  #LSTMTest()
  #Conv2DTest()
  #dropoutTest()
  #poolingTest()
  #EmbeddingTest()
  optimizerTest()
