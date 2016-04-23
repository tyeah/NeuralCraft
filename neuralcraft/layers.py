'''
To use this module, one need to define input tensors and an empty dictionary params = {}
Every layer function receives tensor expressions, params and param names and layer 
definition options defines shared variables and places them in params. return another 
tensor expression(s)
'''
import theano
from theano import tensor as T
import numpy as np

def name_suf(name, suffix):
  if name == None:
    return suffix
  else:
    return name + suffix

def add_param(shape, params, name=None, val=None):
  name = name_suf(name, '_%d' % len(params))
  if isinstance(val, theano.tensor.sharedvar.TensorSharedVariable):
    assert(shape == val.get_value().shape)
    if val.dtype != theano.config.floatX:
      val = val.astype(theano.config.floatX)
    params[name] = val
    return name
  if val is None:
    val = np.random.randn(*shape)
  assert(val.shape == shape)
  params[name] = theano.shared(np.asarray(val, dtype=theano.config.floatX))
  return name

def FCLayer(incoming, params, num_in, num_out, activation=T.nnet.relu, 
    w_name=None, b_name=None, w=None, b=None):
  w_name = name_suf(w_name, 'W_fc')
  b_name = name_suf(b_name, 'b_fc')
  w_name = add_param((num_in, num_out), params, w_name, w)
  b_name = add_param((num_out,), params, b_name, b)
  if incoming.ndim > 2:
    incoming = incoming.flatten(2)
  return activation(incoming.dot(params[w_name]) + params[b_name])

def RNNLayer(incoming, hid_init, params, num_in, num_hidden, activation=T.nnet.relu, 
    w_xh_name=None, w_hh_name=None, b_name=None, w_xh=None, w_hh=None, b=None):
  rnnwxh_name = add_param((num_in, num_hidden), params, w_xh_name, w_xh)
  rnnwhh_name = add_param((num_hidden, num_hidden), params, w_hh_name, w_hh)
  rnnb_name = add_param((num_hidden, ), params, b_name, b)
  def step(income, hid_prev):
    return activation(income.dot(params[rnnwxh_name]) + hid_prev.dot(params[rnnwhh_name]) + params[rnnb_name])
  results, updates = theano.scan(fn=step,
      outputs_info=[{'initial':hid_init, 'taps':[-1]}],
      sequences=[incoming.dimshuffle((1, 0, 2))],
      n_steps=incoming.shape[1])
  return results.dimshuffle((1, 0, 2))

def LSTMLayer(incoming, hid_init, cell_init, params, num_in, num_hidden, activation=T.tanh, gate_act=T.nnet.sigmoid,
    w_xi_name=None, w_hi_name=None, b_i_name=None, w_xi=None, w_hi=None, b_i=None,
    w_xf_name=None, w_hf_name=None, b_f_name=None, w_xf=None, w_hf=None, b_f=None,
    w_xo_name=None, w_ho_name=None, b_o_name=None, w_xo=None, w_ho=None, b_o=None,
    w_xc_name=None, w_hc_name=None, b_c_name=None, w_xc=None, w_hc=None, b_c=None):
  '''
  hid_init and cell_init can be a number, an array or a tensor expression
  '''
  # add parameters
  wxi_name = add_param((num_in, num_hidden), params, w_xi_name, w_xi)
  whi_name = add_param((num_hidden, num_hidden), params, w_hi_name, w_hi)
  bi_name = add_param((num_hidden, ), params, b_i_name, b_i)
  wxf_name = add_param((num_in, num_hidden), params, w_xf_name, w_xf)
  whf_name = add_param((num_hidden, num_hidden), params, w_hf_name, w_hf)
  bf_name = add_param((num_hidden, ), params, b_f_name, b_f)
  wxo_name = add_param((num_in, num_hidden), params, w_xo_name, w_xo)
  who_name = add_param((num_hidden, num_hidden), params, w_ho_name, w_ho)
  bo_name = add_param((num_hidden, ), params, b_o_name, b_o)
  wxc_name = add_param((num_in, num_hidden), params, w_xc_name, w_xc)
  whc_name = add_param((num_hidden, num_hidden), params, w_hc_name, w_hc)
  bc_name = add_param((num_hidden, ), params, b_c_name, b_c)

  # define step function to be used in the loop
  def step(income, hid_prev, cell_prev):
    i = gate_act(income.dot(params[wxi_name]) + hid_prev.dot(params[whi_name]) + params[bi_name])
    f = gate_act(income.dot(params[wxf_name]) + hid_prev.dot(params[whf_name]) + params[bf_name])
    o = gate_act(income.dot(params[wxo_name]) + hid_prev.dot(params[who_name]) + params[bo_name])
    g = activation(income.dot(params[wxc_name]) + hid_prev.dot(params[whc_name]) + params[bc_name])
    cell = f * cell_prev + i * g
    hid = o * activation(cell)
    return [hid, cell]

  # setup hid_init and cell_init
  '''
  if isinstance(hid_init, int) or isinstance(hid_init, float):
    hid_init = hid_init * np.ones((num_hidden,))
  if isinstance(hid_init, np.ndarray):
    #assert hid_init.shape == (1, num_hidden)
    hid_init = theano.shared(hid_init).astype(theano.config.floatX)
  if isinstance(hid_init, theano.tensor.sharedvar.TensorSharedVariable):
    if learn_init:
      add_param(hid_init.get_value().shape, params, hid_init, 'LSTM_hid_init')
  if isinstance(cell_init, int) or isinstance(cell_init, float):
    cell_init = cell_init * np.ones((num_hidden,))
  if isinstance(cell_init, np.ndarray):
    #assert cell_init.shape == (1, num_hidden)
    cell_init = theano.shared(cell_init).astype(theano.config.floatX)
  if isinstance(cell_init, theano.tensor.sharedvar.TensorSharedVariable):
    if learn_init:
      add_param(cell_init.get_value().shape, params, cell_init, 'LSTM_cell_init')
  '''
  if isinstance(hid_init, int) or isinstance(hid_init, float):
    hid_init = hid_init * T.ones((incoming.shape[0], num_hidden))
  if isinstance(hid_init, np.ndarray):
    assert hid_init.shape == (num_hidden, )
    hid_init = np.array(hid_init, dtype=theano.config.floatX)
    hid_init = hid_init * T.ones((incoming.shape[0], num_hidden))
  if isinstance(cell_init, int) or isinstance(cell_init, float):
    cell_init = cell_init * T.ones((incoming.shape[0], num_hidden))
  if isinstance(cell_init, np.ndarray):
    assert cell_init.shape == (num_hidden, )
    cell_init = np.array(cell_init, dtype=theano.config.floatX)
    cell_init = cell_init * T.ones((incoming.shape[0], num_hidden))

  # compose loop
  results, updates = theano.scan(fn=step,
      #outputs_info=[{'initial':[hid_init, cell_init], 'taps':[-1]}],
      outputs_info=[hid_init, cell_init],
      sequences=[incoming.dimshuffle((1, 0, 2))],
      n_steps=incoming.shape[1])
  hid_state, cell_stat = results[0].dimshuffle((1, 0, 2)), results[1].dimshuffle((1, 0, 2))
  return hid_state


def cross_entropy(yhat, y):
  last_dim_len = y.shape[-1]
  if y.ndim == yhat.ndim:
    #y is one-hot
    yhat = T.reshape(-1, last_dim_len)
    y = T.reshape(-1, last_dim_len)
  elif y.ndim == yhat.ndim + 1:
    yhat = T.reshape(-1, last_dim_len)
    y = T.flatten(y)
  return T.mean(T.nnet.categorical_crossentropy(yhatt, yt))
