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
  f = theano.function([xt[0]], xout[0], allow_input_downcast=True) 
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
  print f(x)

def dropoutTest():
  x = np.random.randn(batch_size, seq_size, x_size)

  xt = T.tensor3()
  xt = (xt, x.shape)
  use_noise = T.bscalar()
  yt, _ = dropoutLayer(xt, use_noise, trng, dropout_p)
  f = theano.function([xt[0], use_noise], yt, allow_input_downcast=True) 

  x = np.random.randn(batch_size, seq_size, x_size)
  print f(x, True)

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

def MemTest():
  C = np.random.rand(batch_size, seq_size, embedding_size)
  A = np.random.rand(batch_size, seq_size, embedding_size)
  u = np.random.rand(batch_size, embedding_size)

  Ct = T.tensor3()
  At = T.tensor3()
  ut = T.matrix()
  params = {}

  Ot = MemLayer((ut, u.shape, At, A.shape, Ct, C.shape), params)
  O = theano.function([ut, At, Ct], Ot[0])
  print O(u, A, C)

  def MemNum(u, A, C):
    p = np.tensordot(A, u, axes=([2], [1]))
    O = (C * p).sum(axis=1)
    return O
  print MemNum(u, A, C)

if __name__ == '__main__':
  '''
  FCTest()
  RNNTest()
  LSTMTest()
  Conv2DTest()
  dropoutTest()
  poolingTest()
  EmbeddingTest()
  optimizerTest()
  '''
  MemTest()
