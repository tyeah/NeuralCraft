'''
To use this module, one need to define input tensors and an empty dictionary params = {}
Every layer function receives a tuple of (tensor expressions, input shape), params and param names and layer 
definition options defines shared variables and places them in params. return another 
(tensor expression(s), output_shape)
when defining a net, alwayse use 1 as batch_size, which will not influence the actual net
'''
#TODO: Use astype to define shared variable is problem prone. Fix? Always pass float32 to define?
#TODO: add bias term in conv
#TODO: add init module
#TODO: GRAD_CLIP
import theano
from theano import tensor as T
from theano.tensor import nnet
import numpy as np
from utils import cast_floatX
import init

def name_suf(name, suffix):
  if name == None:
    return suffix
  else:
    return name + suffix

w_initializer = lambda shape: cast_floatX(0.01*np.random.randn(*shape))
'''
def w_initializer(shape):
  print shape
  print cast_floatX(0.01*np.random.randn(*shape))
  return cast_floatX(0.01*np.random.randn(*shape))
  '''
b_initializer = lambda shape: cast_floatX(np.zeros(shape))

def add_param(shape, params, name=None, val=None, initializer=init.HeUniform()):
  if name is None:
    name = name_suf(name, '_%d' % len(params))
  if isinstance(val, theano.tensor.sharedvar.TensorSharedVariable):
    assert(shape == val.get_value().shape)
    assert(val.dtype == theano.config.floatX)
    '''
    if val.dtype != theano.config.floatX:
      val = val.astype(theano.config.floatX)
    '''
    params[name] = val
    return name
  if val is None:
    val = cast_floatX(initializer(shape))
  else:
    val = cast_floatX(val)
  assert(val.shape == shape)
  params[name] = theano.shared(val)
  return name

def FCLayer(incoming, params, num_out, activation=nnet.relu, 
    w_name=None, b_name=None, w=None, b=None, 
    w_initializer=init.HeUniform(), b_initializer=init.Const(0.)):
  incoming, input_shape = incoming
  num_in = np.prod(input_shape[1:])

  output_shape = (input_shape[0], num_out)
  w_name = w_name or 'fc_w_%d' % len(params)
  b_name = b_name or 'b_fc_%d' % len(params)
  w_name = add_param((num_in, num_out), params, w_name, w, w_initializer)
  b_name = add_param((num_out,), params, b_name, b, b_initializer)
  if incoming.ndim > 2:
    incoming = incoming.flatten(2)
  return (activation(T.dot(incoming, params[w_name]) + params[b_name]), output_shape)


def Conv2DLayer(incoming, params, num_out, filter_h, filter_w=None, filter=None, filter_name=None,
    stride_h=None, stride_w=None, padding='half', activation=nnet.relu,
    w_initializer=init.HeUniform(), b_initializer=init.Const(0.)):
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
      params, filter_name or 'conv2d_filter_%d' % len(params), filter, w_initializer)
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
                  #trng.binomial(incoming.shape, p=p, n=1, dtype=theano.config.floatX),
                  incoming * cast_floatX(p))
  #return (proj.astype(theano.config.floatX), output_shape)
  return (proj, output_shape)


def poolingLayer(incoming, ds_h, ds_w=None, stride_h=None, stride_w=None, padding=0, mode='max'):
  '''
  2D pooling
  '''
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


def RNNLayer(incoming, hid_init, params, num_hidden, mask=None, activation=nnet.relu, only_return_final=False,
    w_xh_name=None, w_hh_name=None, b_name=None, w_xh=None, w_hh=None, b=None,
    w_initializer=init.HeUniform(), b_initializer=init.Const(0.)):
  incoming, input_shape = incoming
  num_in = input_shape[-1]

  rnnwxh_name = add_param((num_in, num_hidden), params, w_xh_name or 'rnn_wxh_%d' % len(params), w_xh, w_initializer)
  rnnwhh_name = add_param((num_hidden, num_hidden), params, w_hh_name or 'rnn_whh_%d' % len(params), w_hh, w_initializer)
  rnnb_name = add_param((num_hidden, ), params, b_name or 'rnn_b_%d' % len(params), b, b_initializer)

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
  def step_mask(income, m, hid_prev):
    return T.switch(m, step(income, hid_prev), hid_prev)
  if mask is not None:
    results, updates = theano.scan(fn=step_mask,
        outputs_info=[{'initial':hid_init, 'taps':[-1]}],
        sequences=[incoming.dimshuffle((1, 0, 2)), mask.dimshuffle(1, 0, 'x')])
  else:
    results, updates = theano.scan(fn=step,
        outputs_info=[{'initial':hid_init, 'taps':[-1]}],
        sequences=[incoming.dimshuffle((1, 0, 2))])

  if only_return_final:
    output_shape = (input_shape[0], num_hidden)
    return (results[-1], output_shape)
  else:
    output_shape = (input_shape[0], input_shape[1], num_hidden)
    return (results.dimshuffle((1, 0, 2)), output_shape)


def LSTMLayer_seqfirst_lisa(incoming, cell_init, hid_init, params, num_hidden, 
    W, U, b, W_name='lstm_W', U_name='lstm_U', b_name='lstm_b',
    mask=None, activation=T.tanh, gate_act=nnet.sigmoid, only_return_final=False):
  '''
  hid_init and cell_init can be a number, an array or a tensor expression
  '''
  incoming, input_shape = incoming
  num_in = input_shape[-1]

  # add parameters
  W_name = add_param((num_in, 4 * num_hidden), params, W_name, W)
  U_name = add_param((num_hidden, 4 * num_hidden), params, U_name, U)
  b_name = add_param((4 * num_hidden, ), params, b_name, b)

  def _slice(_x, n, dim):
    if _x.ndim == 3:
      return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]

  # define step function to be used in the loop
  def _step(m_, x_, h_, c_):
    preact = T.dot(h_, params[U_name])
    preact += x_

    i = T.nnet.sigmoid(_slice(preact, 0, num_hidden))
    f = T.nnet.sigmoid(_slice(preact, 1, num_hidden))
    o = T.nnet.sigmoid(_slice(preact, 2, num_hidden))
    c = T.tanh(_slice(preact, 3, num_hidden))

    c = f * c_ + i * c

    h = o * T.tanh(c)

    return h, c

  def _step_mask(m_, x_, h_, c_):
    preact = T.dot(h_, params[U_name])
    preact += x_

    i = T.nnet.sigmoid(_slice(preact, 0, num_hidden))
    f = T.nnet.sigmoid(_slice(preact, 1, num_hidden))
    o = T.nnet.sigmoid(_slice(preact, 2, num_hidden))
    c = T.tanh(_slice(preact, 3, num_hidden))

    c = f * c_ + i * c
    c = m_[:, None] * c + (1. - m_)[:, None] * c_

    h = o * T.tanh(c)
    h = m_[:, None] * h + (1. - m_)[:, None] * h_

    return h, c

  # setup hid_init and cell_init
  if isinstance(hid_init, int) or isinstance(hid_init, float):
    hid_init = hid_init * T.ones((incoming.shape[1], num_hidden))
  if isinstance(hid_init, np.ndarray):
    assert hid_init.shape == (num_hidden, )
    hid_init = np.array(hid_init, dtype=theano.config.floatX)
    hid_init = hid_init * T.ones((incoming.shape[1], num_hidden))
  if isinstance(cell_init, int) or isinstance(cell_init, float):
    cell_init = cell_init * T.ones((incoming.shape[1], num_hidden))
  if isinstance(cell_init, np.ndarray):
    assert cell_init.shape == (num_hidden, )
    cell_init = np.array(cell_init, dtype=theano.config.floatX)
    cell_init = cell_init * T.ones((incoming.shape[1], num_hidden))

  incoming = (T.dot(incoming, params[W_name]) + params[b_name])

  # compose loop
  if mask is not None:
    results, updates = theano.scan(fn=_step_mask,
        outputs_info=[hid_init, cell_init],
        sequences=[mask, incoming])
  else:
    results, updates = theano.scan(fn=_step,
        outputs_info=[hid_init, cell_init],
        sequences=[incoming])
  if only_return_final:
    output_shape = (input_shape[0], num_hidden)
    return (results[0][-1], output_shape)
  else:
    output_shape = (input_shape[0], input_shape[1], num_hidden)
    #cell_stat = results[1].dimshuffle((1, 0, 2))
    hid_state = results[0]
    return (hid_state, output_shape)


def LSTMLayer_seqfirst(incoming, cell_init, hid_init, params, num_hidden, mask=None, activation=T.tanh, gate_act=nnet.sigmoid, only_return_final=False,
    w_xi_name=None, w_hi_name=None, b_i_name=None, w_xi=None, w_hi=None, b_i=None,
    w_xf_name=None, w_hf_name=None, b_f_name=None, w_xf=None, w_hf=None, b_f=None,
    w_xo_name=None, w_ho_name=None, b_o_name=None, w_xo=None, w_ho=None, b_o=None,
    w_xc_name=None, w_hc_name=None, b_c_name=None, w_xc=None, w_hc=None, b_c=None,
    w_initializer=init.HeUniform(), b_initializer=init.Const(0.)):
  '''
  hid_init and cell_init can be a number, an array or a tensor expression
  '''
  incoming, input_shape = incoming
  num_in = input_shape[-1]

  # add parameters
  wxi_name = add_param((num_in, num_hidden), params, w_xi_name or 'lstm_wxi_%d' % len(params), w_xi, w_initializer)
  whi_name = add_param((num_hidden, num_hidden), params, w_hi_name or 'lstm_whi_%d' % len(params), w_hi, w_initializer)
  bi_name = add_param((num_hidden, ), params, b_i_name or 'lstm_bi_%d' % len(params), b_i, b_initializer)
  wxf_name = add_param((num_in, num_hidden), params, w_xf_name or 'lstm_wxf_%d' % len(params), w_xf, w_initializer)
  whf_name = add_param((num_hidden, num_hidden), params, w_hf_name or 'lstm_whf_%d' % len(params), w_hf, w_initializer)
  bf_name = add_param((num_hidden, ), params, b_f_name or 'lstm_bf_%d' % len(params), b_f, b_initializer)
  wxo_name = add_param((num_in, num_hidden), params, w_xo_name or 'lstm_wxo_%d' % len(params), w_xo, w_initializer)
  who_name = add_param((num_hidden, num_hidden), params, w_ho_name or 'lstm_who_%d' % len(params), w_ho, w_initializer)
  bo_name = add_param((num_hidden, ), params, b_o_name or 'lstm_bo_%d' % len(params), b_o, b_initializer)
  wxc_name = add_param((num_in, num_hidden), params, w_xc_name or 'lstm_wxc_%d' % len(params), w_xc, w_initializer)
  whc_name = add_param((num_hidden, num_hidden), params, w_hc_name or 'lstm_whc_%d' % len(params), w_hc, w_initializer)
  bc_name = add_param((num_hidden, ), params, b_c_name or 'lstm_bc_%d' % len(params), b_c, b_initializer)

  def _slice(_x, n, dim):
    if _x.ndim == 3:
      return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]

  wx_concat = T.concatenate((params[wxi_name], params[wxf_name], params[wxo_name], params[wxc_name]), axis=1)
  wh_concat = T.concatenate((params[whi_name], params[whf_name], params[who_name], params[whc_name]), axis=1)
  b_concat = T.concatenate((params[bi_name], params[bf_name], params[bo_name], params[bc_name]), axis=0)

  # define step function to be used in the loop
  def step(income, hid_prev, cell_prev):
    lin_trans = income.dot(wx_concat) + hid_prev.dot(wh_concat) + b_concat
    i = gate_act(_slice(lin_trans, 0, num_hidden))
    f = gate_act(_slice(lin_trans, 1, num_hidden))
    o = gate_act(_slice(lin_trans, 2, num_hidden))
    c = activation(_slice(lin_trans, 3, num_hidden))

    cell = f * cell_prev + i * c
    hid = o * activation(cell)
    return [hid, cell]
  def step_mask(income, m, hid_prev, cell_prev):
    hid, cell = step(income, hid_prev, cell_prev)
    hid = T.switch(m, hid, hid_prev)
    cell = T.switch(m, cell, cell_prev)
    return [hid, cell]

  # setup hid_init and cell_init
  if isinstance(hid_init, int) or isinstance(hid_init, float):
    hid_init = hid_init * T.ones((incoming.shape[1], num_hidden))
  if isinstance(hid_init, np.ndarray):
    assert hid_init.shape == (num_hidden, )
    hid_init = np.array(hid_init, dtype=theano.config.floatX)
    hid_init = hid_init * T.ones((incoming.shape[1], num_hidden))
  if isinstance(cell_init, int) or isinstance(cell_init, float):
    cell_init = cell_init * T.ones((incoming.shape[1], num_hidden))
  if isinstance(cell_init, np.ndarray):
    assert cell_init.shape == (num_hidden, )
    cell_init = np.array(cell_init, dtype=theano.config.floatX)
    cell_init = cell_init * T.ones((incoming.shape[1], num_hidden))

  # compose loop
  if mask is not None:
    results, updates = theano.scan(fn=step_mask,
        outputs_info=[hid_init, cell_init],
        #outputs_info={'initial':[hid_init, cell_init], 'taps':[-1]},
        sequences=[incoming, mask.dimshuffle(0, 1, 'x')])
  else:
    results, updates = theano.scan(fn=step,
        outputs_info=[hid_init, cell_init],
        #outputs_info=[{'initial':[hid_init, cell_init], 'taps':[-1]}],
        sequences=[incoming])
  if only_return_final:
    output_shape = (input_shape[0], num_hidden)
    return (results[0][-1], output_shape)
  else:
    output_shape = (input_shape[0], input_shape[1], num_hidden)
    #cell_stat = results[1].dimshuffle((1, 0, 2))
    hid_state = results[0]
    return (hid_state, output_shape)


def LSTMLayer(incoming, cell_init, hid_init, params, num_hidden, mask=None, activation=T.tanh, gate_act=nnet.sigmoid, only_return_final=False,
    w_xi_name=None, w_hi_name=None, b_i_name=None, w_xi=None, w_hi=None, b_i=None,
    w_xf_name=None, w_hf_name=None, b_f_name=None, w_xf=None, w_hf=None, b_f=None,
    w_xo_name=None, w_ho_name=None, b_o_name=None, w_xo=None, w_ho=None, b_o=None,
    w_xc_name=None, w_hc_name=None, b_c_name=None, w_xc=None, w_hc=None, b_c=None,
    w_initializer=init.HeUniform(), b_initializer=init.Const(0.)):
  '''
  hid_init and cell_init can be a number, an array or a tensor expression
  '''
  incoming, input_shape = incoming
  num_in = input_shape[-1]

  # add parameters
  wxi_name = add_param((num_in, num_hidden), params, w_xi_name or 'lstm_wxi_%d' % len(params), w_xi, w_initializer)
  whi_name = add_param((num_hidden, num_hidden), params, w_hi_name or 'lstm_whi_%d' % len(params), w_hi, w_initializer)
  bi_name = add_param((num_hidden, ), params, b_i_name or 'lstm_bi_%d' % len(params), b_i, b_initializer)
  wxf_name = add_param((num_in, num_hidden), params, w_xf_name or 'lstm_wxf_%d' % len(params), w_xf, w_initializer)
  whf_name = add_param((num_hidden, num_hidden), params, w_hf_name or 'lstm_whf_%d' % len(params), w_hf, w_initializer)
  bf_name = add_param((num_hidden, ), params, b_f_name or 'lstm_bf_%d' % len(params), b_f, b_initializer)
  wxo_name = add_param((num_in, num_hidden), params, w_xo_name or 'lstm_wxo_%d' % len(params), w_xo, w_initializer)
  who_name = add_param((num_hidden, num_hidden), params, w_ho_name or 'lstm_who_%d' % len(params), w_ho, w_initializer)
  bo_name = add_param((num_hidden, ), params, b_o_name or 'lstm_bo_%d' % len(params), b_o, b_initializer)
  wxc_name = add_param((num_in, num_hidden), params, w_xc_name or 'lstm_wxc_%d' % len(params), w_xc, w_initializer)
  whc_name = add_param((num_hidden, num_hidden), params, w_hc_name or 'lstm_whc_%d' % len(params), w_hc, w_initializer)
  bc_name = add_param((num_hidden, ), params, b_c_name or 'lstm_bc_%d' % len(params), b_c, b_initializer)

  def _slice(_x, n, dim):
    if _x.ndim == 3:
      return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]

  wx_concat = T.concatenate((params[wxi_name], params[wxf_name], params[wxo_name], params[wxc_name]), axis=1)
  wh_concat = T.concatenate((params[whi_name], params[whf_name], params[who_name], params[whc_name]), axis=1)
  b_concat = T.concatenate((params[bi_name], params[bf_name], params[bo_name], params[bc_name]), axis=0)

  # define step function to be used in the loop
  def step(income, hid_prev, cell_prev):
    lin_trans = income.dot(wx_concat) + hid_prev.dot(wh_concat) + b_concat
    i = gate_act(_slice(lin_trans, 0, num_hidden))
    f = gate_act(_slice(lin_trans, 1, num_hidden))
    o = gate_act(_slice(lin_trans, 2, num_hidden))
    c = activation(_slice(lin_trans, 3, num_hidden))

    cell = f * cell_prev + i * c
    hid = o * activation(cell)
    return [hid, cell]
  def step_mask(income, m, hid_prev, cell_prev):
    hid, cell = step(income, hid_prev, cell_prev)
    hid = T.switch(m, hid, hid_prev)
    cell = T.switch(m, cell, cell_prev)
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
  if mask is not None:
    results, updates = theano.scan(fn=step_mask,
        outputs_info=[hid_init, cell_init],
        #outputs_info={'initial':[hid_init, cell_init], 'taps':[-1]},
        sequences=[incoming.dimshuffle((1, 0, 2)), mask.dimshuffle(1, 0, 'x')])
  else:
    results, updates = theano.scan(fn=step,
        outputs_info=[hid_init, cell_init],
        #outputs_info=[{'initial':[hid_init, cell_init], 'taps':[-1]}],
        sequences=[incoming.dimshuffle((1, 0, 2))])
  if only_return_final:
    output_shape = (input_shape[0], num_hidden)
    return (results[0][-1], output_shape)
  else:
    output_shape = (input_shape[0], input_shape[1], num_hidden)
    #cell_stat = results[1].dimshuffle((1, 0, 2))
    hid_state = results[0].dimshuffle((1, 0, 2))
    return (hid_state, output_shape)


def EmbeddingLayer(incoming, params, num_in, num_out, w_name=None, w=None, initializer=init.HeUniform()):
  '''
  input a (batch of) iscalar i, output the corresponding embedding vector, which
  is, the ith row of embedding matrix w.
  num_in is the number of possible inputs (upper bound of i, vocabulary size)
  '''
  incoming, input_shape = incoming
  output_shape = (input_shape[0], input_shape[1], num_out)

  w_name = add_param((num_in, num_out), params, w_name or 'emb_%d' % len(params), w, initializer)

  return (params[w_name][incoming], output_shape)
  #return (params[w_name][incoming.flatten()].reshape([
    #incoming.shape[0], incoming.shape[1], num_out]), output_shape)
