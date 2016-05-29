import theano
from theano import tensor as T

def l2(params):
  return T.sum([T.sum(p**2) for p in params.values()])

def l1(params):
  return T.sum([T.sum(T.abs(p)) for p in params.values()])
