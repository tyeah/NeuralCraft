import theano
from theano import tensor as T
import numpy as np
from utils import cast_floatX

'''
def sgd(cost, params, lr=1e-2):
  grads = T.grad(cost, params.values())
  updates = []
  for p, g in zip(params.values(), grads):
    updates.append([p, p - g * lr])
  return updates

'''
def sgd(cost, incomings, params, lr=1e-2):
  '''
  #incomings should be a list
  '''
  grads = T.grad(cost, params.values())
  updates = []
  for p, g in zip(params.values(), grads):
    updates.append([p, p - g * lr])
  if isinstance(lr, T.TensorVariable):
    return theano.function(incomings + [lr], cost, updates=updates, allow_input_downcast=True)
  else:
    return theano.function(incomings, cost, updates=updates, allow_input_downcast=True)

def momentum(cost, incomings, params, lr=1e-2, mu=0.9):
  '''
  incomings should be a list
  '''
  grads = T.grad(cost, params.values())
  velocity = [theano.shared(cast_floatX(np.zeros_like(p.get_value()))) for p in params.values()]
  updates = []
  def v_update(v, g):
    return mu * v - lr * g
  for p, g, v in zip(params.values(), grads, velocity):
    # since theano will update only based on p'value before p is updated, 
    # we need to explicitly express the update of p
    updates.append([p, p + v_update(v, g)])
    updates.append([v, v_update(v, g)])
  if isinstance(lr, T.TensorVariable):
    return theano.function(incomings + [lr], cost, updates=updates, allow_input_downcast=True)
  else:
    return theano.function(incomings, cost, updates=updates, allow_input_downcast=True)

def nesterov_momentum(cost, incomings, params, lr=1e-2, mu=0.9):
  '''
  incomings should be a list
  '''
  grads = T.grad(cost, params.values())
  velocity = [theano.shared(cast_floatX(np.zeros_like(p.get_value()))) for p in params.values()]
  updates = []
  def v_update(v, g):
    return mu * v - lr * g
  for p, g, v in zip(params.values(), grads, velocity):
    updates.append([p, p - mu * v + (1 + mu) * v_update(v, g)])
    updates.append([v, v_update(v, g)])
  if isinstance(lr, T.TensorVariable):
    return theano.function(incomings + [lr], cost, updates=updates, allow_input_downcast=True)
  else:
    return theano.function(incomings, cost, updates=updates, allow_input_downcast=True)

def adagrad(cost, incomings, params, lr=1e-2, epsilon=1e-8):
  '''
  incomings should be a list
  '''
  grads = T.grad(cost, params.values())
  cache = [theano.shared(cast_floatX(np.zeros_like(p.get_value()))) for p in params.values()]
  updates = []
  def c_update(c, g):
    return c + T.sqr(g)
  for p, g, c in zip(params.values(), grads, cache):
    updates.append([p, p - lr * g / (np.sqrt(c_update(c, g)) + epsilon)])
    updates.append([c, c_update(c, g)])
  if isinstance(lr, T.TensorVariable):
    return theano.function(incomings + [lr], cost, updates=updates, allow_input_downcast=True)
  else:
    return theano.function(incomings, cost, updates=updates, allow_input_downcast=True)

def rmsprop(cost, incomings, params, lr=1e-2, dr=0.99, epsilon=1e-8):
  '''
  incomings should be a list
  '''
  grads = T.grad(cost, params.values())
  cache = [theano.shared(cast_floatX(np.zeros_like(p.get_value()))) for p in params.values()]
  updates = []
  def c_update(c, g):
    return dr * c + (1 - dr) * T.sqr(g)
  for p, g, c in zip(params.values(), grads, cache):
    updates.append([p, p - lr * g / (np.sqrt(c_update(c, g)) + epsilon)])
    updates.append([c, c_update(c, g)])
  if isinstance(lr, T.TensorVariable):
    return theano.function(incomings + [lr], cost, updates=updates, allow_input_downcast=True)
  else:
    return theano.function(incomings, cost, updates=updates, allow_input_downcast=True)

def adam(cost, incomings, params, lr=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8):
  '''
  incomings should be a list
  '''
  grads = T.grad(cost, params.values())
  velocity = [theano.shared(cast_floatX(np.zeros_like(p.get_value()))) for p in params.values()]
  momentum = [theano.shared(cast_floatX(np.zeros_like(p.get_value()))) for p in params.values()]
  beta1_run = theano.shared(cast_floatX(beta1))
  beta2_run = theano.shared(cast_floatX(beta2))
  # cannot use beta2_run = theano.shared(beta2).astype(theano.config.floatX)
  # since the target of updates in theano.function has to be an shared, not an Elemwise{Cast{float32}}.0
  updates = []
  def m_update(m, g):
    return beta1 * m + (1 - beta1) * g
  def mb(m):
    return m / (1 - beta1_run)
    #return m
  def v_update(v, g):
    return beta2 * v + (1 - beta2) * T.sqr(g)
  def vb(v):
    return v / (1 - beta2_run)
    #return v
  for p, g, v, m in zip(params.values(), grads, velocity, momentum):
    # since theano will update only based on p'value before p is updated, 
    # we need to explicitly express the update of p
    updates.append([p, p - lr * mb(m_update(m, g)) / (T.sqrt(vb(v_update(v, g))) + epsilon)])
    updates.append([v, v_update(v, g)])
    updates.append([m, m_update(m, g)])
  updates.append([beta1_run, beta1_run * beta1])
  updates.append([beta2_run, beta2_run * beta2])
  if isinstance(lr, T.TensorVariable):
    return theano.function(incomings + [lr], cost, updates=updates, allow_input_downcast=True)
  else:
    return theano.function(incomings, cost, updates=updates, allow_input_downcast=True)
