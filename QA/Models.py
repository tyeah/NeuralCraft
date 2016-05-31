import pickle

import theano
from theano import tensor as T

from neuralcraft import layers, optimizers, utils, init, regularizer


def model(model_name):
    models = {'lstm': LSTM_Model,
              'memn2n': MemN2N_Model,
              'bi-lstm': Bidirectional_LSTM,
              'att-lstm': Attention_LSTM}

    return models[model_name]


class Model(object):
    def __init__(self, options):
        self.oo = options['optimization_options']
        self.mo = options['model_options']
        self.optimizer = optimizers.get_optimizer(self.oo['optimizer'])
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
        n_hops = self.mo['n_hops']

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
        net['u_emb'] = layers.EmbeddingLayer(u_in, self.params, vs, es, w_name='E')
        net['u_lstm'] = layers.LSTMLayer(net['u_emb'],
                                         0.,
                                         0.,
                                         self.params,
                                         nh,
                                         umaskt,
                                         only_return_final=True)
        net['c_emb'] = layers.EmbeddingLayer(c_in, self.params, vs, es, w_name='E')
        net['c_emb_rsp'] = layers.ReshapeLayer(net['c_emb'],
                                               ('x', cl * sl, es))
        cmaskt_rsp = cmaskt.reshape((-1, cl * sl))
        for i in range(n_hops):
            if i == 0:
                net['c_lstm_%d' % i] = layers.LSTMLayer(net['c_emb_rsp'],
                                                 0.,
                                                 0.,
                                                 self.params,
                                                 nh,
                                                 cmaskt_rsp)
            else:
                net['c_lstm_%d_in' % i] = layers.ConcatLayer(
                        (net['c_emb_rsp'], net['c_lstm_%d'%(i-1)]), axis=2)
                net['c_lstm_%d'%i] = layers.LSTMLayer(net['c_lstm_%d_in'%i],
                                                 0.,
                                                 0.,
                                                 self.params,
                                                 nh,
                                                 cmaskt_rsp)
        c_lstm_concat = tuple([(net['c_lstm_%d'%i][0][:, -1, :], ('x', nh))  for i in range(n_hops)])
        net['concat'] = layers.ConcatLayer(
            c_lstm_concat+(net['u_lstm'],), axis=1)
        net['output'] = layers.FCLayer(net['concat'],
                                       self.params,
                                       vs,
                                       activation=T.nnet.softmax)
        pred_prob = theano.function(
            [ct, cmaskt, ut, umaskt],
            net['output'][0], allow_input_downcast=True)
        pred = theano.function([ct, cmaskt, ut, umaskt],
                               net['output'][0].argmax(axis=1),
                               allow_input_downcast=True)

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
        n_hops = self.mo['n_hops']
        pe = self.mo['position_encode']
        te = self.mo['temporal_encode']

        u_shape = ('x', sl)
        c_shape = ('x', cl, sl)
        a_shape = ('x', )

        ut = T.imatrix()
        umaskt = T.bmatrix()
        ct = T.itensor3()
        cmaskt = T.btensor3()
        at = T.ivector()

        lr = T.scalar()
        opt_options = {'lr': lr, 'clip_norm': 40}

        u_in = (ut, u_shape)
        c_in = (ct, c_shape)
        a_in = (at, a_shape)
        self.linear = theano.shared(1.)
        self.use_noise = theano.shared(1.)

        if pe:
            J = sl
            d = es
            PE_row = T.arange(1, J + 1, dtype=theano.config.floatX)
            PE_column = T.arange(1, d + 1, dtype=theano.config.floatX)
            PE = (1.0 - PE_row / J)[:, None] - (PE_column / d)[None, :] * (
                1.0 - 2 * PE_row / J)[:, None]
        B_name = 'A' if self.mo['AB_share'] else 'B'

        net = {}
        for nh in range(n_hops):
            if nh == 0:
                net['u_emb_%d' % nh] = layers.EmbeddingLayer(
                    u_in,
                    self.params,
                    vs,
                    es,
                    w_name=B_name,
                    initializer=init.Gaussian(sigma=0.1))
                if pe:
                    net['u_emb_%d' % nh] = list(net['u_emb_%d' % nh])
                    net['u_emb_%d' % nh][0] *= PE[None, :, :]
            #     '''
            # if self.oo['dropout']:
            #   net['u_emb_%d' % nh] = layers.DropoutLayer(net['u_emb_%d' % nh], self.use_noise, self.oo['p_dropout'])
            # '''
                net['u_emb_%d' % nh] = (net['u_emb_%d' % nh][0] *
                                        umaskt[:, :, None], net['u_emb_%d' %
                                                                nh][1])
                net['u_combine_%d' % nh] = layers.SumLayer(
                    net['u_emb_%d' % nh], axis=1)
                net['a_emb_%d' % nh] = layers.EmbeddingLayer(
                    c_in,
                    self.params,
                    vs,
                    es,
                    w_name='A',
                    initializer=init.Gaussian(sigma=0.1))
                if pe:
                    net['a_emb_%d' % nh] = list(net['a_emb_%d' % nh])
                    net['a_emb_%d' % nh][0] *= PE[None, None, :, :]
            #     '''
            # if self.oo['dropout']:
            #   net['a_emb_%d' % nh] = layers.DropoutLayer(net['a_emb_%d' % nh], self.use_noise, self.oo['p_dropout'])
            # '''
                net['a_emb_%d' % nh] = (net['a_emb_%d' % nh][0] *
                                        cmaskt[:, :, :, None],
                                        net['a_emb_%d' % nh][1])
                net['a_combine_%d' % nh] = layers.SumLayer(
                    net['a_emb_%d' % nh], axis=2)
                if te:
                    net['a_combine_%d' % nh] = layers.TemporalEncodeLayer(
                        net['a_combine_%d' % nh],
                        self.params,
                        T_name='T_a')
                net['c_emb_%d' % nh] = layers.EmbeddingLayer(
                    c_in,
                    self.params,
                    vs,
                    es,
                    w_name='C',
                    initializer=init.Gaussian(sigma=0.1))
                if pe:
                    net['c_emb_%d' % nh] = list(net['c_emb_%d' % nh])
                    net['c_emb_%d' % nh][0] *= PE[None, None, :, :]
            #     '''
            # if self.oo['dropout']:
            #   net['c_emb_%d' % nh] = layers.DropoutLayer(net['c_emb_%d' % nh], self.use_noise, self.oo['p_dropout'])
            # '''
                net['c_emb_%d' % nh] = (net['c_emb_%d' % nh][0] *
                                        cmaskt[:, :, :, None],
                                        net['c_emb_%d' % nh][1])
                net['c_combine_%d' % nh] = layers.SumLayer(
                    net['c_emb_%d' % nh], axis=2)
                if te:
                    net['c_combine_%d' % nh] = layers.TemporalEncodeLayer(
                        net['c_combine_%d' % nh],
                        self.params,
                        T_name='T_c')

                net['o_%d' % nh], net['attention_%d' % nh] = layers.MemLayer(
                    (net['u_combine_%d' % nh], net['a_combine_%d' % nh],
                     net['c_combine_%d' % nh]), self.params, self.linear)
            else:
                net['u_combine_%d' %
                    nh] = layers.LinearLayer(net['u_combine_%d' % (nh - 1)],
                                             self.params,
                                             es,
                                             w_name='H')
                net['u_combine_%d' % nh] = layers.ElementwiseCombineLayer(
                    (net['u_combine_%d' % nh], net['o_%d' % (nh - 1)]), T.add)
                net['a_emb_%d' % nh] = layers.EmbeddingLayer(
                    c_in, self.params, vs,
                    es, w_name='A')
                if pe:
                    net['a_emb_%d' % nh] = list(net['a_emb_%d' % nh])
                    net['a_emb_%d' % nh][0] *= PE[None, None, :, :]
            #     '''
            # if self.oo['dropout']:
            #   net['a_emb_%d' % nh] = layers.DropoutLayer(net['a_emb_%d' % nh], self.use_noise, self.oo['p_dropout'])
            # '''
                net['a_emb_%d' % nh] = (net['a_emb_%d' % nh][0] *
                                        cmaskt[:, :, :, None],
                                        net['a_emb_%d' % nh][1])
                net['a_combine_%d' % nh] = layers.SumLayer(
                    net['a_emb_%d' % nh], axis=2)
                if te:
                    net['a_combine_%d' % nh] = layers.TemporalEncodeLayer(
                        net['a_combine_%d' % nh],
                        self.params,
                        T_name='T_a')
                net['c_emb_%d' % nh] = layers.EmbeddingLayer(
                    c_in, self.params, vs,
                    es, w_name='C')
                if pe:
                    net['c_emb_%d' % nh] = list(net['c_emb_%d' % nh])
                    net['c_emb_%d' % nh][0] *= PE[None, None, :, :]
            #     '''
            # if self.oo['dropout']:
            #   net['c_emb_%d' % nh] = layers.DropoutLayer(net['c_emb_%d' % nh], self.use_noise, self.oo['p_dropout'])
            # '''
                net['c_emb_%d' % nh] = (net['c_emb_%d' % nh][0] *
                                        cmaskt[:, :, :, None],
                                        net['c_emb_%d' % nh][1])
                net['c_combine_%d' % nh] = layers.SumLayer(net['c_emb_%d' %
                                                               nh],
                                                           axis=2)
                if te:
                    net['c_combine_%d' % nh] = layers.TemporalEncodeLayer(
                        net['c_combine_%d' % nh],
                        self.params,
                        T_name='T_c')

                net['o_%d' % nh], net['attention_%d' % nh] = layers.MemLayer(
                    (net['u_combine_%d' % nh], net['a_combine_%d' % nh],
                     net['c_combine_%d' % nh]), self.params, self.linear)

        net['ou'] = layers.ElementwiseCombineLayer(
            (net['o_%d' % nh], net['u_combine_%d' % nh]), T.add)
        if self.oo['dropout']:
            net['ou'] = layers.DropoutLayer(net['ou'], self.use_noise,
                                            self.oo['p_dropout'])

        net['output'] = layers.LinearLayer(
            net['ou'],
            self.params,
            vs,
            activation=T.nnet.softmax,
            w_name='w_fc',
            w_initializer=init.Gaussian(sigma=0.1))

        pred_prob = theano.function(
            [ct, cmaskt, ut, umaskt],
            net['output'][0], allow_input_downcast=True)
        pred = theano.function([ct, cmaskt, ut, umaskt],
                               net['output'][0].argmax(axis=1),
                               allow_input_downcast=True)

        cost = T.nnet.categorical_crossentropy(net['output'][0], at).mean()

        if self.oo['reg'] == 'l2':
            cost += self.oo['reg_weight'] * regularizer.l2(self.params)
        elif self.oo['reg'] == 'l1':
            cost += self.oo['reg_weight'] * regularizer.l1(self.params)

        update = self.optimizer(cost, [ct, cmaskt, ut, umaskt, at],
                                self.params, opt_options)

        attention = [theano.function(
            [ct, cmaskt, ut, umaskt],
            net['attention_%d' % nh][0], allow_input_downcast=True)
                     for nh in range(n_hops)]

        self.pred_prob, self.pred, self.update, self.attention = pred_prob, pred, update, attention


class Bidirectional_LSTM(Model):
    def __init__(self, options):
        super(Bidirectional_LSTM, self).__init__(options)

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
        net['u_emb'] = layers.EmbeddingLayer(u_in, self.params, vs, es, w_name='E')
        net['u_lstm'] = layers.LSTMLayer(net['u_emb'],
                                         0.,
                                         0.,
                                         self.params,
                                         nh,
                                         umaskt,
                                         only_return_final=True)
        net['u_lstm_rev'] = layers.LSTMLayer(
            (net['u_emb'][0][:, ::-1, :], net['u_emb'][1]),
            0.,
            0.,
            self.params,
            nh,
            umaskt[:, ::-1],
            only_return_final=True)
        net['u_lstm_bidir'] = layers.ConcatLayer(
            (net['u_lstm'],
             net['u_lstm_rev']), axis=1)
        net['c_emb'] = layers.EmbeddingLayer(c_in, self.params, vs, es, w_name='E')
        net['c_emb_rsp'] = layers.ReshapeLayer(net['c_emb'],
                                               ('x', cl * sl, es))
        cmaskt_rsp = cmaskt.reshape((-1, cl * sl))
        net['c_lstm'] = layers.LSTMLayer(net['c_emb_rsp'],
                                         0.,
                                         0.,
                                         self.params,
                                         nh,
                                         cmaskt_rsp)
        net['c_lstm_rev'] = layers.LSTMLayer(
            (net['c_emb_rsp'][0][:, ::-1, :], net['c_emb_rsp'][1]),
            0.,
            0.,
            self.params,
            nh,
            cmaskt_rsp[:, ::-1])
        net['c_lstm_bidir'] = layers.ConcatLayer(
            (net['c_lstm'],
             net['c_lstm_rev']), axis=2)
        net['c_lstm_slice'] = layers.SliceLayer(
            net['c_lstm_bidir'], axis=1, step=sl)
        net['c_lstm_mean'] = layers.MeanLayer(net['c_lstm_slice'], axis=1)

        net['concat'] = layers.ConcatLayer(
            (net['u_lstm_bidir'], net['c_lstm_mean']),
            axis=1)
        net['output'] = layers.FCLayer(net['concat'],
                                       self.params,
                                       vs,
                                       activation=T.nnet.softmax)
        pred_prob = theano.function(
            [ct, cmaskt, ut, umaskt],
            net['output'][0], allow_input_downcast=True)
        pred = theano.function([ct, cmaskt, ut, umaskt],
                               net['output'][0].argmax(axis=1),
                               allow_input_downcast=True)

        cost = T.nnet.categorical_crossentropy(net['output'][0], at).mean()

        update = self.optimizer(cost, [ct, cmaskt, ut, umaskt, at],
                                self.params, opt_options)

        self.pred_prob, self.pred, self.update = pred_prob, pred, update

class Attention_LSTM(Model):
    def __init__(self, options):
        super(Attention_LSTM, self).__init__(options)

    def build(self):
        vs = self.mo['vocab_size']
        es = self.mo['embedding_size']
        ms = self.mo['m_size']
        gs = self.mo['g_size']
        nh = self.mo['num_hid']
        sl = self.mo['sentence_length']
        cl = self.mo['context_length']
        sla = self.mo['sentence_level_att']

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

        self.use_noise = theano.shared(1.)

        net = {}
        net['u_emb'] = layers.EmbeddingLayer(u_in, self.params, vs, es, w_name='E')
        net['u_lstm'] = layers.LSTMLayer(net['u_emb'],
                                         0.,
                                         0.,
                                         self.params,
                                         nh,
                                         umaskt,
                                         only_return_final=True)
        net['u_lstm_rev'] = layers.LSTMLayer(
            (net['u_emb'][0][:, ::-1, :], net['u_emb'][1]),
            0.,
            0.,
            self.params,
            nh,
            umaskt[:, ::-1],
            only_return_final=True)
        net['u_lstm_bidir'] = layers.ConcatLayer(
            (net['u_lstm'],
             net['u_lstm_rev']), axis=1)
        net['c_emb'] = layers.EmbeddingLayer(c_in, self.params, vs, es, w_name='E')
        net['c_emb_rsp'] = layers.ReshapeLayer(net['c_emb'],
                                               ('x', cl * sl, es))
        cmaskt_rsp = cmaskt.reshape((-1, cl * sl))
        net['c_lstm'] = layers.LSTMLayer(net['c_emb_rsp'],
                                         0.,
                                         0.,
                                         self.params,
                                         nh,
                                         cmaskt_rsp)
        net['c_lstm_rev'] = layers.LSTMLayer(
            (net['c_emb_rsp'][0][:, ::-1, :], net['c_emb_rsp'][1]),
            0.,
            0.,
            self.params,
            nh,
            cmaskt_rsp[:, ::-1])
        if sla:
            net['c_lstm_slice_forward'] = layers.SliceLayer(
                net['c_lstm'], axis=1, step=sl, start=sl-1)
            net['c_lstm_slice_backward'] = layers.SliceLayer(
                net['c_lstm_rev'], axis=1, step=sl, start=0)
            net['c_lstm_slice'] = layers.ConcatLayer(
                    (net['c_lstm_slice_forward'], net['c_lstm_slice_backward']), axis=2)
            if self.oo['dropout']:
                net['u_lstm_bidir'] = layers.DropoutLayer(net['u_lstm_bidir'], self.use_noise,
                                                self.oo['p_dropout'])
                net['c_lstm_slice'] = layers.DropoutLayer(net['c_lstm_slice'], self.use_noise,
                                                self.oo['p_dropout'])
            net['Wu'] = layers.LinearLayer(net['u_lstm_bidir'], self.params, ms)
            net['Wc'] = layers.LinearLayer(net['c_lstm_slice'], self.params, ms)
            broadcast_u = net['Wu'][0][:,None,:], net['Wc'][1]
            net['Wu+Wc'] = layers.ElementwiseCombineLayer((broadcast_u, net['Wc']))
            net['m'] = T.tanh(net['Wu+Wc'][0]), net['Wu+Wc'][1]
            net['wTm'] = layers.ReshapeLayer(
                         layers.LinearLayer(net['m'], self.params, 1), ('x', cl))
            s, sshape = T.nnet.softmax(net['wTm'][0]), net['wTm'][1]
            net['r'] = T.sum(s[:, :, None] * net['c_lstm_slice'][0], axis=1), \
                       ('x', nh * 2)
        else:
            net['c_lstm_bidir'] = layers.ConcatLayer(
                (net['c_lstm'],
                 net['c_lstm_rev']), axis=2)
            if self.oo['dropout']:
                net['u_lstm_bidir'] = layers.DropoutLayer(net['u_lstm_bidir'], self.use_noise,
                                                self.oo['p_dropout'])
                net['c_lstm_bidir'] = layers.DropoutLayer(net['c_lstm_bidir'], self.use_noise,
                                                self.oo['p_dropout'])
            net['Wu'] = layers.LinearLayer(net['u_lstm_bidir'], self.params, ms)
            net['Wc'] = layers.LinearLayer(net['c_lstm_bidir'], self.params, ms)
            broadcast_u = net['Wu'][0][:,None,:], net['Wc'][1]
            net['Wu+Wc'] = layers.ElementwiseCombineLayer((broadcast_u, net['Wc']))
            net['m'] = T.tanh(net['Wu+Wc'][0]), net['Wu+Wc'][1]
            net['wTm'] = layers.ReshapeLayer(
                         layers.LinearLayer(net['m'], self.params, 1), ('x', cl * sl))
            s, sshape = T.nnet.softmax(net['wTm'][0]), net['wTm'][1]
            net['r'] = T.sum(s[:, :, None] * net['c_lstm_bidir'][0], axis=1), \
                       ('x', nh * 2)

        net['ru'] = layers.ConcatLayer((net['r'], net['u_lstm_bidir']), axis=1)
        if self.oo['dropout']:
            net['ru'] = layers.DropoutLayer(net['ru'], self.use_noise,
                                            self.oo['p_dropout'])
        net['g'] = layers.LinearLayer(net['ru'], self.params, gs, activation=T.tanh)
        if self.oo['dropout']:
            net['g'] = layers.DropoutLayer(net['g'], self.use_noise,
                                            self.oo['p_dropout'])

        net['output'] = layers.FCLayer(net['g'],
                                       self.params,
                                       vs,
                                       activation=T.nnet.softmax)
        pred_prob = theano.function(
            [ct, cmaskt, ut, umaskt],
            net['output'][0], allow_input_downcast=True)
        pred = theano.function([ct, cmaskt, ut, umaskt],
                               net['output'][0].argmax(axis=1),
                               allow_input_downcast=True)

        cost = T.nnet.categorical_crossentropy(net['output'][0], at).mean()

        if self.oo['reg'] == 'l2':
            cost += self.oo['reg_weight'] * regularizer.l2(self.params)
        elif self.oo['reg'] == 'l1':
            cost += self.oo['reg_weight'] * regularizer.l1(self.params)

        update = self.optimizer(cost, [ct, cmaskt, ut, umaskt, at],
                                self.params, opt_options)

        self.pred_prob, self.pred, self.update = pred_prob, pred, update
