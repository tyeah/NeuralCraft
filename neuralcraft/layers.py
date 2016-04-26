'''
To use this module, one need to define input tensors and an empty dictionary params = {}
Every layer function receives a tuple of (tensor expressions, input shape), params and param names and layer 
definition options defines shared variables and places them in params. return another 
(tensor expression(s), output_shape)
'''
import theano
from theano import tensor as T
from theano.tensor import nnet
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

def FCLayer(incoming, params, num_out, activation=nnet.relu, 
    w_name=None, b_name=None, w=None, b=None):
  incoming, input_shape = incoming
  num_in = np.prod(input_shape[1:])

  output_shape = (input_shape[0], num_out)
  w_name = name_suf(w_name, 'W_fc')
  b_name = name_suf(b_name, 'b_fc')
  w_name = add_param((num_in, num_out), params, w_name, w)
  b_name = add_param((num_out,), params, b_name, b)
  if incoming.ndim > 2:
    incoming = incoming.flatten(2)
  return (activation(incoming.dot(params[w_name]) + params[b_name]), output_shape)


def Conv2DLayer(incoming, params, num_out, filter_h, filter_w=None, filter=None, filter_name=None,
    stride_h=None, stride_w=None, padding='half', activation=nnet.relu):
  '''
  incoming shoule be a tensor4: (batch_size, channel_size, height, width)
  filter should be None or ndarray or shared
  here num_in == channel_size. how to infer automatically?
  ''' 
  incoming, input_shape = incoming
  num_in, input_h, input_w = input_shape[-3:]

  assert filter_h % 2 == 1
  if not filter_w:
    filter_w = filter_h
  if not stride_h:
    stride_h = 1
  if not stride_w:
    stride_w = stride_h
  assert filter==None or \
      (isinstance(filter, np.ndarray) and \
      filter.shape==(num_out, incoming.shape[1], filter_h, filter_w))\
      or (isinstance(filter, theano.tensor.sharedvar.TensorSharedVariable) and \
      filter.get_value().shape==(num_out, incoming.shape[1], filter_h, filter_w))
  filter_name = add_param((num_out, num_in, filter_h, filter_w), 
      params, filter_name, filter)
  if padding == 'half':
    output_h, output_w = input_h, input_w
  else:
    raise NotImplementedError("not implemented output shape for padding patterns other than 'half'")
  output_shape = (input_shape[0], num_out, output_h, output_w)
  output = activation(nnet.conv2d(incoming, params[filter_name], border_mode=padding, 
    subsample=(stride_h, stride_w)))
  return (output, output_shape)


def dropoutLayer(incoming, use_noise, trng, p):
  """
  tensor switch is like an if statement that checks the
  value of the theano shared variable (use_noise) (we can also use 0/1), 
  before either dropping out the incoming tensor or
  computing the appropriate activation. During training/testing
  use_noise is toggled on and off.
  """
  incoming, input_shape = incoming
  output_shape = input_shape
  proj = T.switch(use_noise,
                  incoming *
                  trng.binomial(incoming.shape, p=p, n=1, dtype=incoming.dtype),
                  incoming * p)
  return (proj, output_shape)


def poolingLayer(incoming, ds_h, ds_w=None, stride_h=None, stride_w=None, padding=0, mode='max'):
  incoming, input_shape = incoming
  input_h, input_w = input_shape[-2:]
  if not ds_w:
    ds_w = ds_h
  ds = (ds_w, ds_h)
  if not stride_h and not stride_w:
    st = ds
  elif stride_h != None and stride_w != None:
    st = (stride_h, stride_w)
  else:
    st = stride_h if stride != None else stride_w
    st = (st, st)
  if isinstance(padding, int):
    padding = (padding, padding)
  elif isinstance(padding, tuple):
    assert len(padding) == 2
  output = T.signal.pool.pool_2d(incoming, ds, ignore_border=False, st=st, padding=padding, mode=mode)
  output_h = input_h + 2 * padding[0]
  output_h = int(output_h / st[0])
  output_w = input_w + 2 * padding[1]
  output_w = int(output_w / st[1])
  output_shape = tuple(list(input_shape[:-2]) + [output_h, output_w])
  return output, output_shape


def RNNLayer(incoming, hid_init, params, num_hidden, activation=nnet.relu, 
    w_xh_name=None, w_hh_name=None, b_name=None, w_xh=None, w_hh=None, b=None):
  incoming, input_shape = incoming
  num_in = input_shape[-1]
  output_shape = (input_shape[0], input_shape[1], num_hidden)

  rnnwxh_name = add_param((num_in, num_hidden), params, w_xh_name, w_xh)
  rnnwhh_name = add_param((num_hidden, num_hidden), params, w_hh_name, w_hh)
  rnnb_name = add_param((num_hidden, ), params, b_name, b)

  # setup hid_init
  if isinstance(hid_init, int) or isinstance(hid_init, float):
    hid_init = hid_init * T.ones((incoming.shape[0], num_hidden))
  if isinstance(hid_init, np.ndarray):
    assert hid_init.shape == (num_hidden, )
    hid_init = np.array(hid_init, dtype=theano.config.floatX)
    hid_init = hid_init * T.ones((incoming.shape[0], num_hidden))

  # setup step function
  def step(income, hid_prev):
    return activation(income.dot(params[rnnwxh_name]) + hid_prev.dot(params[rnnwhh_name]) + params[rnnb_name])
  results, updates = theano.scan(fn=step,
      outputs_info=[{'initial':hid_init, 'taps':[-1]}],
      sequences=[incoming.dimshuffle((1, 0, 2))],
      n_steps=incoming.shape[1])
  return (results.dimshuffle((1, 0, 2)), output_shape)


def LSTMLayer(incoming, cell_init, hid_init, params, num_hidden, activation=T.tanh, gate_act=nnet.sigmoid,
    w_xi_name=None, w_hi_name=None, b_i_name=None, w_xi=None, w_hi=None, b_i=None,
    w_xf_name=None, w_hf_name=None, b_f_name=None, w_xf=None, w_hf=None, b_f=None,
    w_xo_name=None, w_ho_name=None, b_o_name=None, w_xo=None, w_ho=None, b_o=None,
    w_xc_name=None, w_hc_name=None, b_c_name=None, w_xc=None, w_hc=None, b_c=None):
  '''
  hid_init and cell_init can be a number, an array or a tensor expression
  '''
  incoming, input_shape = incoming
  num_in = input_shape[-1]
  output_shape = (input_shape[0], input_shape[1], num_hidden)

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
  return (hid_state, output_shape)


def EmbeddingLayer(incoming, params, num_in, num_out, w_name=None, w=None):
  '''
  input a (batch of) iscalar i, output the corresponding embedding vector, which
  is, the ith row of embedding matrix w.
  num_in is the number of possible inputs (upper bound of i, vocabulary size)
  '''
  incoming, input_shape = incoming
  output_shape = (input_shape[0], num_out)

  w_name = add_param((num_in, num_out), params, w_name, w)

  return (params[w_name][incoming], output_shape)


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