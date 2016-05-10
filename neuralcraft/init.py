import theano
from theano import tensor as T
from theano.tensor import nnet
import numpy as np
from utils import cast_floatX

w_initializer = lambda shape: cast_floatX(0.01*np.random.randn(*shape))
b_initializer = lambda shape: cast_floatX(np.zeros(shape))

class Initializer(object):
  def __call__(self, shape):
    return self.sample(shape)

  def sample(self, shape):
    raise NotImplementedError()


class Const(Initializer): 
  def __init__(self, val=0.):
    self.val = val

  def sample(self, shape):
    return cast_floatX(np.ones(shape) * self.val)


class Gaussian(Initializer):
  def __init__(self, mu=0., sigma=1.):
    self.mu = mu
    self.sigma = sigma

  def sample(self, shape):
    return cast_floatX(self.sigma * (np.random.randn(*shape) + self.mu))


class Uniform(Initializer):
  def __init__(self, range=0.01, sigma=None, mu=0.):
    if sigma != None:
      a = mu - np.sqrt(3) * sigma
      b = mu + np.sqrt(3) * sigma
    else:
      try:
        a, b = range
      except TypeError:
        a, b = -range, range
    self.a, self.b = (a, b)

  def sample(self, shape):
    return cast_floatX(np.random.rand(*shape) * (self.b - self.a) + self.a)


class He(Initializer):
  def __init__(self, initializer, gain=1.0, c01b=False):
    if gain == 'relu':
      gain=np.sqrt(2)

    self.initializer = initializer
    self.gain = gain
    self.c01b = c01b

  def sample(self, shape):
    if self.c01b:
      assert len(shape) == 4, 'when c01b==True, shape must have length 4'
      fan_in = np.prod(shape[:3])
    else:
      assert len(shape) >= 2, 'shape must have length >= 2'
      if len(shape) == 2:
        fan_in = shape[0]
      else:
        fn_in = np.prod(shape[1:])

    sigma = self.gain * np.sqrt(1.0 / fan_in)
    return self.initializer(sigma=sigma).sample(shape)


class HeGaussian(He):
  def __init__(self, gain=1.0, c01b=False):
    super(HeGaussian, self).__init__(Gaussian, gain, c01b)


class HeUniform(He):
  def __init__(self, gain=1.0, c01b=False):
    super(HeUniform, self).__init__(Uniform, gain, c01b)


class Orth(Initializer):
  def __init__(self, gain=1.0):
    if gain == 'relu':
      gain = np.sqrt(2)
    self.gain = gain

  def sample(self, shape):
    assert len(shape) >= 2, 'shape must have length >= 2'
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.randn(*flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q
