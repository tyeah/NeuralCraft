from neuralcraft import layers, optimizers, utils
import theano
from theano import tensor as T

class model(object):
  def __init__(self, options):
    self.oo = options['optimization_options']
    self.mo = options['model_options']
    if self.oo['optimizer'] == 'sgd':
      self.optimizer = optimizers.sgd
    elif self.oo['optimizer'] == 'rmsprop':
      self.optimizer = optimizers.rmsprop

  def build(self):
    if self.mo['model_name'] == 'lstm':
      self.pred_preb, self.pred, self.update = self.LSTM()
    elif self.mo['model_name'] == 'memn2n':
      self.pred_preb, self.pred, self.update = self.MemN2N()

  def LSTM(self):
    vs = self.mo['vocab_size']
    es = self.mo['embedding_size']
    nh = self.mo['num_hid']
    sl = self.mo['sentence_length']
    cl = self.mo['context_length']

    u_shape = ('x', sl)
    c_shape = ('x', cl, sl)
    a_shape = ('x',)

    ut = T.imatrix()
    umaskt = T.bmatrix()
    ct = T.itensor3()
    cmaskt = T.btensor3()
    at = T.ivector()

    lr = T.scalar()
    opt_options = {'lr': lr}

    u_in = (ut, u_shape)
    c_in = (ct, c_shape)
    a_in = (at, a_shape)

    params = {}
    net = {}
    net['u_emb'] = layers.EmbeddingLayer(u_in, params, vs, es)
    net['u_lstm'] = layers.LSTMLayer(net['u_emb'], 0., 0., params, nh, umaskt, only_return_final=True)
    net['c_emb'] = layers.EmbeddingLayer(c_in, params, vs, es)
    net['c_emb_rsp'] = layers.ReshapeLayer(net['c_emb'], ('x', cl * sl, es))
    cmaskt_rsp = cmaskt.reshape((-1, cl * sl))
    net['c_lstm'] = layers.LSTMLayer(net['c_emb_rsp'], 0., 0., params, nh, cmaskt_rsp, only_return_final=True)
    net['concat'] = layers.ConcatLayer((net['c_lstm'], net['u_lstm']), axis=1)
    net['output'] = layers.FCLayer(net['concat'], params, vs, activation=T.nnet.softmax)
    pred_prob = theano.function([ct, cmaskt, ut, umaskt], net['output'][0])
    pred = theano.function([ct, cmaskt, ut, umaskt], net['output'][0].argmax(axis=1))

    cost = T.nnet.categorical_crossentropy(net['output'][0], at).mean()

    update = self.optimizer(cost, [ct, cmaskt, ut, umaskt, at], params, opt_options)

    return pred_prob, pred, update

  def MemN2N(self):
    vs = self.mo['vocab_size']
    es = self.mo['embedding_size']
    nh = self.mo['num_hid']
    sl = self.mo['sentence_length']
    cl = self.mo['context_length']
    #n_hops = self.mo['n_hops']

    u_shape = ('x', sl)
    c_shape = ('x', cl, sl)
    a_shape = ('x',)

    ut = T.imatrix()
    umaskt = T.bmatrix()
    ct = T.itensor3()
    cmaskt = T.btensor3()
    at = T.ivector()

    lr = T.scalar()
    opt_options = {'lr': lr}

    u_in = (ut, u_shape)
    c_in = (ct, c_shape)
    a_in = (at, a_shape)

    params = {}
    net = {}
    net['u_emb'] = layers.EmbeddingLayer(u_in, params, vs, es, w_name='B')
    net['u_emb']= (net['u_emb'][0] * umaskt[:, :, None], net['u_emb'][1])
    net['u_combine'] = layers.SumLayer(net['u_emb'], axis=1)
    #net['u_combine'] = (net['u_combine'][0] / T.sum(umaskt, axis=1)[:, None], net['u_combine'][1]) #optional
    net['a_emb'] = layers.EmbeddingLayer(c_in, params, vs, es, w_name='A')
    net['a_emb']= (net['a_emb'][0] * cmaskt[:, :, :,  None], net['a_emb'][1])
    net['a_combine'] = layers.SumLayer(net['a_emb'], axis=2)
    #net['a_combine'] = (net['a_combine'][0] / T.sum(cmaskt, axis=2)[:, None], net['a_combine'][1]) #optional
    net['c_emb'] = layers.EmbeddingLayer(c_in, params, vs, es, w_name='C')
    net['c_emb']= (net['c_emb'][0] * cmaskt[:, :, :,  None], net['c_emb'][1])
    net['c_combine'] = layers.SumLayer(net['c_emb'], axis=2)
    #net['c_combine'] = (net['c_combine'][0] / T.sum(cmaskt, axis=2)[:, None], net['c_combine'][1]) #optional

    net['o'] = layers.MemLayer((net['u_combine'], net['a_combine'], net['c_combine']), params)
    net['ou'] = layers.ElementwiseCombineLayer((net['o'], net['u_combine']), T.add)
    net['output'] = layers.FCLayer(net['ou'], params, vs, activation=T.nnet.softmax, w_name='w_fc', b_name='b_fc')

    pred_prob = theano.function([ct, cmaskt, ut, umaskt], net['output'][0])
    pred = theano.function([ct, cmaskt, ut, umaskt], net['output'][0].argmax(axis=1))

    cost = T.nnet.categorical_crossentropy(net['output'][0], at).mean()

    update = self.optimizer(cost, [ct, cmaskt, ut, umaskt, at], params, opt_options)

    return pred_prob, pred, update
