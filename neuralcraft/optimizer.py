import theano
from theano import tensor as T
import numpy as np

def sgd(cost, params, lr=1e-2):
  grads = T.grad(cost, params.values())
  updates = []
  for p, g in zip(params.values(), grads):
    updates.append([p, p - g * lr])
  return updates

