import pickle

import theano
from theano import tensor as T

from neuralcraft import layers, optimizers, utils


def model(model_name):
    models = {'lstm': LSTM_Model, 'memn2n': MemN2N_Model, }

    return models[model_name]


class Model(object):
    def __init__(self, options):
        self.oo = options['optimization_options']
        self.mo = options['model_options']
        if self.oo['optimizer'] == 'sgd':
            self.optimizer = optimizers.sgd
        elif self.oo['optimizer'] == 'rmsprop':
            self.optimizer = optimizers.rmsprop
        self.params = {}

    def load_params(self, path):
        print "load from %s" % path
        params_val = pickle.load(open(path, 'r'))
        assert set(params_val.keys()) == set(self.params.keys(
        )), 'not compatible params'
        for k, v in params_val.iteritems():
            self.params[k].set_value(v)

    def dump_params(self, path):
        params_val = {}
        for k, v in self.params.iteritems():
            params_val[k] = v.get_value()
        pickle.dump(params_val, open(path, 'w'))
        print "\ndump to %s\n" % path

    def build(self):
        raise NotImplementedError()


class LSTM_Model(Model):
    def __init__(self, options):
        super(LSTM_Model, self).__init__(options)

    def build(self):
        vs = self.mo['vocab_size']
        es = self.mo['embedding_size']
        nh = self.mo['num_hid']
        sl = self.mo['sentence_length']
        cl = self.mo['context_length']

        u_shape = ('x', sl)
        c_shape = ('x', cl, sl)
        a_shape = ('x', )

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

        net = {}
        net['u_emb'] = layers.EmbeddingLayer(u_in, self.params, vs, es)
        net['u_lstm'] = layers.LSTMLayer(net['u_emb'],
                                         0.,
                                         0.,
                                         self.params,
                                         nh,
                                         umaskt,
                                         only_return_final=True)
        net['c_emb'] = layers.EmbeddingLayer(c_in, self.params, vs, es)
        net['c_emb_rsp'] = layers.ReshapeLayer(net['c_emb'],
                                               ('x', cl * sl, es))
        cmaskt_rsp = cmaskt.reshape((-1, cl * sl))
        net['c_lstm'] = layers.LSTMLayer(net['c_emb_rsp'],
                                         0.,
                                         0.,
                                         self.params,
                                         nh,
                                         cmaskt_rsp,
                                         only_return_final=True)
        net['concat'] = layers.ConcatLayer(
            (net['c_lstm'],
             net['u_lstm']), axis=1)
        net['output'] = layers.FCLayer(net['concat'],
                                       self.params,
                                       vs,
                                       activation=T.nnet.softmax)
        pred_prob = theano.function(
            [ct, cmaskt, ut, umaskt],
            net['output'][0], allow_input_downcast=True)
        pred = theano.function([ct, cmaskt, ut, umaskt],
                               net['output'][0].argmax(
                                   axis=1), allow_input_downcast=True)

        cost = T.nnet.categorical_crossentropy(net['output'][0], at).mean()

        update = self.optimizer(cost, [ct, cmaskt, ut, umaskt, at],
                                self.params, opt_options)

        self.pred_prob, self.pred, self.update = pred_prob, pred, update


class MemN2N_Model(Model):
    def __init__(self, options):
        super(MemN2N_Model, self).__init__(options)

    def build(self):
        vs = self.mo['vocab_size']
        es = self.mo['embedding_size']
        nh = self.mo['num_hid']
        sl = self.mo['sentence_length']
        cl = self.mo['context_length']
        #n_hops = self.mo['n_hops']

        u_shape = ('x', sl)
        c_shape = ('x', cl, sl)
        a_shape = ('x', )

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

        net = {}
        net['u_emb'] = layers.EmbeddingLayer(
            u_in, self.params, vs, es, w_name='B')
        net['u_emb'] = (net['u_emb'][0] * umaskt[:, :, None], net['u_emb'][1])
        net['u_combine'] = layers.SumLayer(net['u_emb'], axis=1)
        #net['u_combine'] = (net['u_combine'][0] / T.sum(umaskt[:, None].astype(theano.config.floatX), axis=1), net['u_combine'][1]) #optional
        net['a_emb'] = layers.EmbeddingLayer(
            c_in, self.params, vs, es, w_name='A')
        net['a_emb'] = (net['a_emb'][0] * cmaskt[:, :, :, None],
                        net['a_emb'][1])
        net['a_combine'] = layers.SumLayer(net['a_emb'], axis=2)
        #net['a_combine'] = (net['a_combine'][0] / T.sum(cmaskt[:, :, None].astype(theano.config.floatX), axis=2), net['a_combine'][1]) #optional
        net['c_emb'] = layers.EmbeddingLayer(
            c_in, self.params, vs, es, w_name='C')
        net['c_emb'] = (net['c_emb'][0] * cmaskt[:, :, :, None],
                        net['c_emb'][1])
        net['c_combine'] = layers.SumLayer(net['c_emb'], axis=2)
        #net['c_combine'] = (net['c_combine'][0] / T.sum(cmaskt[:, :, None].astype(theano.config.floatX), axis=2), net['c_combine'][1]) #optional

        net['o'] = layers.MemLayer(
            (net['u_combine'], net['a_combine'],
             net['c_combine']), self.params)
        net['ou'] = layers.ElementwiseCombineLayer(
            (net['o'], net['u_combine']), T.add)
        net['output'] = layers.FCLayer(net['ou'],
                                       self.params,
                                       vs,
                                       activation=T.nnet.softmax,
                                       w_name='w_fc',
                                       b_name='b_fc')

        pred_prob = theano.function(
            [ct, cmaskt, ut, umaskt],
            net['output'][0], allow_input_downcast=True)
        pred = theano.function([ct, cmaskt, ut, umaskt],
                               net['output'][0].argmax(
                                   axis=1), allow_input_downcast=True)

        cost = T.nnet.categorical_crossentropy(net['output'][0], at).mean()

        update = self.optimizer(cost, [ct, cmaskt, ut, umaskt, at],
                                self.params, opt_options)

        self.pred_prob, self.pred, self.update = pred_prob, pred, update
