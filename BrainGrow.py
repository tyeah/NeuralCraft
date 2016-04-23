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
  #assert False, 'not implemented'
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
  def step(income, hid_prev, cell_prev):
    i = gate_act(income.dot(params[wxi_name]) + hid_prev.dot(params[whi_name]) + params[bi_name])
    f = gate_act(income.dot(params[wxf_name]) + hid_prev.dot(params[whf_name]) + params[bf_name])
    o = gate_act(income.dot(params[wxo_name]) + hid_prev.dot(params[who_name]) + params[bo_name])
    g = activation(income.dot(params[wxc_name]) + hid_prev.dot(params[whc_name]) + params[bc_name])
    cell = f * cell_prev + i * g
    hid = o * activation(cell)
    return [hid, cell]
  results, updates = theano.scan(fn=step,
      #outputs_info=[{'initial':[hid_init, cell_init], 'taps':[-1]}],
      outputs_info=[hid_init, cell_init],
      sequences=[incoming.dimshuffle((1, 0, 2))],
      n_steps=incoming.shape[1])
  return results[0].dimshuffle((1, 0, 2))


def sgd(cost, params, lr=1e-2):
  grads = T.grad(cost, params.values())
  updates = []
  for p, g in zip(params.values(), grads):
    updates.append([p, p - g * lr])
  return updates

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



if __name__ == '__main__':
  '''
  # FC test
  xt = T.matrix()
  yt = T.ivector()
  params = {}
  xout = FCLayer(xt, params, 3, 2)
  yhatt = T.nnet.softmax(xout)
  loss = T.mean(T.nnet.categorical_crossentropy(yhatt, yt))
  updates = sgd(loss, params)

  f = theano.function([xt, yt], loss, updates = updates, allow_input_downcast=True) 

  x = np.random.randn(5, 3)
  y = np.array([0, 1, 1, 0, 0])
  for i in range(10):
    print f(x, y)
    print [v.get_value() for v in params.values()]

  # RNN test
  xt = T.tensor3()
  hidt = T.matrix()
  yt = T.imatrix()
  params = {}
  rnnout = RNNLayer(xt, hidt, params, 3, 4)
  rnnout = rnnout.reshape((-1, rnnout.shape[-1]))
  # neet to reshape flatten first 2 dim of rnn output before inputing to FCLayer
  yhatt = FCLayer(rnnout, params, 4, 2, activation=T.nnet.softmax)
  loss = T.mean(T.nnet.categorical_crossentropy(yhatt, yt.flatten()))
  updates = sgd(loss, params)

  f = theano.function([xt, hidt, yt], loss, updates = updates, allow_input_downcast=True) 

  x = np.random.randn(2, 5, 3)
  hid = np.random.randn(2, 4)
  y = np.array([[0, 1, 1, 0, 1], [1, 0, 0, 0, 1]])
  for i in range(10):
    print f(x, hid, y)
    print [v.get_value() for v in params.values()]
  '''
  #LSTM test
  xt = T.tensor3()
  hidt = T.matrix()
  cellt = T.matrix()
  yt = T.imatrix()
  params = {}
  rnnout = LSTMLayer(xt, hidt, cellt, params, 3, 4)
  rnnout = rnnout.reshape((-1, rnnout.shape[-1]))
  # neet to reshape flatten first 2 dim of rnn output before inputing to FCLayer
  yhatt = FCLayer(rnnout, params, 4, 2, activation=T.nnet.softmax)
  loss = T.mean(T.nnet.categorical_crossentropy(yhatt, yt.flatten()))
  updates = sgd(loss, params)

  f = theano.function([xt, hidt, cellt, yt], loss, updates = updates, allow_input_downcast=True) 

  x = np.random.randn(2, 5, 3)
  hid = np.random.randn(2, 4)
  cell = np.random.randn(2, 4)
  y = np.array([[0, 1, 1, 0, 1], [1, 0, 0, 0, 1]])
  for i in range(10):
    print f(x, hid, cell, y)
    print [v.get_value() for v in params.values()]
