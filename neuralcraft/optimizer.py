import theano
from theano import tensor as T
import numpy as np

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
  incomings should be a list
  '''
  '''
  return theano.function(incomings, cost, allow_input_downcast=True)
  '''
  grads = T.grad(cost, params.values())
  updates = []
  for p, g in zip(params.values(), grads):
    updates.append([p, p - g * lr])
  print updates
  if isinstance(lr, T.TensorVariable):
    return theano.function(incomings + [lr], cost, updates)
  else:
    return theano.function(incomings, cost, updates)
