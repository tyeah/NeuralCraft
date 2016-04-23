import theano
import numpy as np

class Layer(object):
  def __init__(self, **kwargs):
    self.parameters = {}
    self.parameter_names = []

  def add_param(self, shape, name=None, val=None):
    if name == None:
      name = 'W_%d' % len(self.parameters) 
    if isinstance(val, theano.tensor.sharedvar.TensorSharedVariable):
      assert(shape == val.get_value().shape)
      if val.dtype != theano.config.floatX:
        val = val.astype(theano.config.floatX)
      self.parameters[name] = val
      return
    if val is None:
      val = np.random.randn(*shape)
    assert(val.shape == shape)
    self.parameters[name] = theano.shared(np.asarray(val, dtype=theano.config.floatX))

  def parameter_names(self):
    self.parameters_names = self.parameters.keys()
    return self.parameters_names;


class FullyConnectedLayer(Layer):
  def __init__(self, num_in, num_out, W=None, **kwargs):
    super(FullyConnectedLayer, self).__init__(**kwargs)
    self.add_param((num_in, num_out), 'W', W)

