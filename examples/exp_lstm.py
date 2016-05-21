import theano
from theano import tensor as T
from theano.tensor import nnet
import numpy as np
from neuralcraft.utils import cast_floatX
from neuralcraft.layers import LSTMLayer, RNNLayer
import lasagne

batch_size = 3
seq_len = 4
hid_size = 2
x_size = 2

w_xi = cast_floatX(np.random.randn(x_size, hid_size))
w_xf = cast_floatX(np.random.randn(x_size, hid_size))
w_xo = cast_floatX(np.random.randn(x_size, hid_size))
w_xc = cast_floatX(np.random.randn(x_size, hid_size))

w_hi = cast_floatX(np.random.randn(hid_size, hid_size))
w_hf = cast_floatX(np.random.randn(hid_size, hid_size))
w_ho = cast_floatX(np.random.randn(hid_size, hid_size))
w_hc = cast_floatX(np.random.randn(hid_size, hid_size))

bi = cast_floatX(np.zeros(hid_size))
bf = cast_floatX(np.zeros(hid_size))
bo = cast_floatX(np.zeros(hid_size))
bc = cast_floatX(np.zeros(hid_size))

x = cast_floatX(np.random.randn(batch_size, seq_len, x_size))
x_shape = x.shape
mask = np.random.rand(batch_size, seq_len) > 0.5
mask_shape = mask.shape

xt = T.tensor3()
maskt = T.bmatrix()

'---------------------------------------------------------------------------'
# with mask
params = {}
lstm0t, _ = LSTMLayer((xt, x_shape), 0, 0, params, hid_size, maskt,
    w_xi=w_xi, w_xf=w_xf, w_xo=w_xo, w_xc=w_xc,
    w_hi=w_xi, w_hf=w_xf, w_ho=w_xo, w_hc=w_xc,
    b_i=bi, b_f=bf, b_o=bo, b_c=bc)
#lstm0 = theano.function([xt, maskt], lstm0t, allow_input_downcast=True)
loss0t = lstm0t.sum()
grads0t = T.grad(loss0t, params.values())
#grads0t = T.grad(loss0t, params['lstm_wxc_9'])
#print params.keys()
grads0 = theano.function([xt, maskt], grads0t, allow_input_downcast=True)

lin = lasagne.layers.InputLayer(x_shape, xt)
lmask = lasagne.layers.InputLayer(mask_shape, maskt)
llstm = lasagne.layers.LSTMLayer(lin, hid_size, mask_input=lmask, peepholes=False,
    ingate=lasagne.layers.Gate(W_in=w_xi, W_hid=w_hi, b=bi, W_cell=lasagne.init.Constant(0.)),
    forgetgate=lasagne.layers.Gate(W_in=w_xf, W_hid=w_hf, b=bf, W_cell=lasagne.init.Constant(0.)),
    cell=lasagne.layers.Gate(W_in=w_xc, W_hid=w_hc, b=bc, W_cell=lasagne.init.Constant(0.), 
      nonlinearity=lasagne.nonlinearities.tanh),
    outgate=lasagne.layers.Gate(W_in=w_xo, W_hid=w_ho, b=bo, W_cell=lasagne.init.Constant(0.)))
lstm1t = lasagne.layers.get_output(llstm)
#lstm1 = theano.function([xt, maskt], lstm1t, allow_input_downcast=True)
loss1t = lstm1t.sum()
params_lasagne = lasagne.layers.get_all_params(llstm, trainable=True)
print params_lasagne
grads1t = T.grad(loss1t, params_lasagne)
grads1 = theano.function([xt, maskt], grads1t, allow_input_downcast=True)

print 'lstm0_neuralcraft'
#print lstm0(x, mask)
grads0_val = grads0(x, mask)
for k, v in enumerate(params.keys()):
  print params.keys()[k]
  print grads0_val[k]

print 'lstm1_lasagne'
#print lstm1(x, mask)
#print grads1(x, mask)
grads1_val = grads1(x, mask)
for k, v in enumerate(params_lasagne):
  print params_lasagne[k]
  print grads1_val[k]

def lstm_py(x, mask):
  x = x.transpose(1, 0, 2)
  mask = mask.transpose(1, 0)
  def sigmoid(y):
    return 1 / (1 + np.exp(-y))
  def step(xx, hid_prev, cell_prev):
    i = sigmoid(xx.dot(w_xi) + hid_prev.dot(w_hi) + bi)
    f = sigmoid(xx.dot(w_xf) + hid_prev.dot(w_hf) + bf)
    o = sigmoid(xx.dot(w_xo) + hid_prev.dot(w_ho) + bo)
    c = np.tanh(xx.dot(w_xc) + hid_prev.dot(w_hc) + bc)
    cell = f * cell_prev + i * c
    hid = o * np.tanh(cell)
    return [hid, cell]
  def step_mask(xx, m, hid_prev, cell_prev):
    hid, cell = step(xx, hid_prev, cell_prev)
    m = m.reshape(-1, 1).astype('float32')
    return (m * hid + (1-m) * hid_prev, m * cell + (1-m) * cell_prev)

  hid_prev = np.zeros((batch_size, hid_size))
  cell_prev = np.zeros((batch_size, hid_size))
  hids = np.zeros((seq_len, batch_size, hid_size))
  cells = np.zeros((seq_len, batch_size, hid_size))
  for i in range(seq_len):
    hid, cell = step_mask(x[i], mask[i], hid_prev, cell_prev)
    hids[i], cells[i] = hid, cell
    hid_prev, cell_prev = hid, cell
  return hids.transpose(1, 0, 2)

#print 'lstm2_python'
#print lstm_py(x, mask)
'---------------------------------------------------------------------------'
'''

'---------------------------------------------------------------------------'
# without mask
params = {}
lstm0, _ = LSTMLayer((xt, x_shape), 0, 0, params, hid_size,
    w_xi=w_xi, w_xf=w_xf, w_xo=w_xo, w_xc=w_xc,
    w_hi=w_xi, w_hf=w_xf, w_ho=w_xo, w_hc=w_xc,
    b_i=bi, b_f=bf, b_o=bo, b_c=bc)
lstm0 = theano.function([xt], lstm0, allow_input_downcast=True)

lin = lasagne.layers.InputLayer(x_shape, xt)
llstm = lasagne.layers.LSTMLayer(lin, hid_size, peepholes=False,
    ingate=lasagne.layers.Gate(W_in=w_xi, W_hid=w_hi, b=bi, W_cell=lasagne.init.Constant(0.)),
    forgetgate=lasagne.layers.Gate(W_in=w_xf, W_hid=w_hf, b=bf, W_cell=lasagne.init.Constant(0.)),
    cell=lasagne.layers.Gate(W_in=w_xc, W_hid=w_hc, b=bc, W_cell=lasagne.init.Constant(0.), 
      nonlinearity=lasagne.nonlinearities.tanh),
    outgate=lasagne.layers.Gate(W_in=w_xo, W_hid=w_ho, b=bo, W_cell=lasagne.init.Constant(0.)))
lstm1 = lasagne.layers.get_output(llstm)
lstm1 = theano.function([xt], lstm1, allow_input_downcast=True)

print 'lstm0_neuralcraft'
print lstm0(x)
print 'lstm1_lasagne'
print lstm1(x)

def lstm_py(x):
  x = x.transpose(1, 0, 2)
  def sigmoid(y):
    return 1 / (1 + np.exp(-y))
  def step(xx, hid_prev, cell_prev):
    i = sigmoid(xx.dot(w_xi) + hid_prev.dot(w_hi) + bi)
    f = sigmoid(xx.dot(w_xf) + hid_prev.dot(w_hf) + bf)
    o = sigmoid(xx.dot(w_xo) + hid_prev.dot(w_ho) + bo)
    c = np.tanh(xx.dot(w_xc) + hid_prev.dot(w_hc) + bc)
    cell = f * cell_prev + i * c
    hid = o * np.tanh(cell)
    return [hid, cell]

  hid_prev = np.zeros((batch_size, hid_size))
  cell_prev = np.zeros((batch_size, hid_size))
  hids = np.zeros((seq_len, batch_size, hid_size))
  cells = np.zeros((seq_len, batch_size, hid_size))
  for i in range(seq_len):
    hid, cell = step(x[i], hid_prev, cell_prev)
    hids[i], cells[i] = hid, cell
    hid_prev, cell_prev = hid, cell
  return hids.transpose(1, 0, 2)

print 'lstm2_python'
print lstm_py(x)
'---------------------------------------------------------------------------'
'''

'''
'---------------------------------------------------------------------------'
# RNN
w_xh = w_xi
w_hh = w_hi
b = bi
params = {}
rnn0, _ = RNNLayer((xt, x_shape), 0, params, hid_size, maskt,
    w_xh=w_xh, w_hh=w_hh, b=b)
rnn0 = theano.function([xt, maskt], rnn0, allow_input_downcast=True)

lrnn = lasagne.layers.RecurrentLayer(lin, hid_size, mask_input=lmask,
    W_hid_to_hid=w_hh, b=b, W_in_to_hid=w_xh)
rnn1 = lasagne.layers.get_output(lrnn)
rnn1 = theano.function([xt, maskt], rnn1, allow_input_downcast=True)

print 'rnn0'
print rnn0(x, mask)
print 'rnn1'
print rnn1(x, mask)
'---------------------------------------------------------------------------'
'''
