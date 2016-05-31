# TODO: weight sharing
# TODO: why randomly sampling quetion doesn't work?
# TODO: Position Encoding!
import os
import argparse
from pathlib2 import Path
import json
from time import time

import numpy as np

from QAReader import QAReader
from minibatch import MinibatchReader
import Models

def read_dataset(data_path, task_num, lang, Reader, extra_options):
    bAbI_base = Path(data_path)
    data_base = bAbI_base / lang
    if 1 <= task_num <= 20:
        try:
            files = os.listdir(str(data_base))
        except OSError:
            raise ValueError('Invalid path: %s' % str(data_base))
        # "train" is lexicographically behind "test"
        test_fn, train_fn = tuple(map(lambda fn: str(data_base / fn), sorted(
            filter(lambda fn: fn.startswith('qa%d_' % task_num), files))))
        train_set = Reader(train_fn, **extra_options)
        test_set = Reader(test_fn,                       dictionaries=train_set.getDictionaries(), **extra_options)
        # This below recreates the situation of experiment.py
        # test_set = Reader(train_fn, dictionaries=train_set.getDictionaries())
    else:
        raise ValueError('Invalid task number: %d' % task_num)
    return train_set, test_set


class QATask(object):
    def __init__(self, options):
        self.do = options['data_options']
        self.mo = options['model_options']
        self.oo = options['optimization_options']
        self.lo = options['log_options']

        data_path = self.do['data_path']
        task_num = self.do['task_number']
        lang = self.do.get('language', 'en')  # defaults to use small Eng set
        self.qa_train, self.qa_test \
            = read_dataset(data_path,
                           task_num, lang, options['data_options']['reader'],
                           {'threshold': 0,
                            'context_length': self.mo['context_length'],
                            'sentence_length': self.mo['sentence_length']})

        self.data_size = len(self.qa_train.stories)

        tokens = self.qa_train.specialWords
        self.NULL = tokens['<NULL>']
        self.EOS = tokens['<EOS>']
        self.UNKNOWN = tokens['<UNKNOWN>']

        if self.oo['dump_params']:
            weight_dir = Path(self.oo['weight_path'])
            if not weight_dir.exists():
                weight_dir.mkdir()
        self.batch_size_train = self.oo['batch_size_train']
        self.batch_size_test = self.oo['batch_size_test']

        self.verbose = self.oo['verbose']
        self.log = self.logger_factory()
        self.lo['dump_epoch'] = self.oo['max_epoch'] \
                                if self.lo['dump_epoch'] < 0 \
                                else self.lo['dump_epoch']

        vocab_size = len(self.qa_train.index_to_word)
        options['model_options']['vocab_size'] = vocab_size
        model_name = self.mo['model_name']
        self.model = Models.model(model_name)(options)

    def logger_factory(self):
        def log(s):
            print s

        def dummy(s):
            pass

        return log if self.verbose else dummy

    def build_model(self):
        self.log('Starting compiling...')
        t0 = time()
        self.model.build()
        if self.oo['load_params']:
            weight_file = os.path.join(self.oo['weight_path'],
                                       self.oo['load_name'])
            try:
                self.model.load_params(weight_file)
            except IOError:
                raise IOError(
                    "The directory specified by weight_path doesn't exist! Please check config file. Set load_params to 'false' if this there is no previously saved weight file.")
        self.log('Ending compiling...\nCompiling time: %.2f' % (time() - t0))

    def train(self):
        max_epoch = self.oo['max_epoch']
        iters_in_epoch = int(self.data_size / self.oo['batch_size_train'])
        lr = self.oo['learning_rate']
        disp_iter = self.oo['disp_iter']
        train_batch = self.qa_train.minibatch(                                   self.oo['batch_size_train'], self.oo['shuffle'])
        test_batch = self.qa_test.minibatch(self.oo['batch_size_test'],
                                    self.oo['shuffle'])
        print 'Starting training...'
        epoch_idx = 0
        iter_idx = 0
        cost_acc = 0
        #test_acc = 0
        train_acc = 0
        test_acc_history = []
        if self.oo['dropout']:
            self.model.use_noise.set_value(1)
        if 'linear_start' in self.oo.keys() and not self.oo['linear_start']:
            self.model.linear.set_value(0.)
        while (True):
            c, cmask, u, umask, a, evidence = next(train_batch)
            #print u
            #train_acc += np.mean(self.model.pred(c, cmask, u, umask) == a)
            '''
            if iter_idx % 120 == 0:
                #print evidence, [np.argmax(att(c, cmask, u, umask), axis=1) for att in self.model.attention]
                att_val = [att(c, cmask, u, umask) for att in self.model.attention]
                att_label_val = np.array([[val[i][evidence[i]] for i in range(len(evidence))] for val in att_val])
                print evidence, [np.argmax(val, axis=1) for val in att_val]#, att_label_val
                #print [np.sum(av < 0.01) for av in att_val]
                #print (cmask.sum(axis=-1) == 0).sum()
            '''
            cost = self.model.update(c, cmask, u, umask, a, lr)
            cost_acc += cost
            '''
            if iter_idx % disp_iter == 0:
                print 'cost at epoch %d, iteration %d: %f' % (epoch_idx,
                                                              iter_idx, cost)
            '''
            iter_idx += 1
            dump_file = os.path.join(self.oo['weight_path'],
                                     self.oo['dump_name'])
            if iter_idx % iters_in_epoch == 0:
                if epoch_idx > 0 and epoch_idx % self.lo["dump_epoch"] == 0:
                    self.model.dump_params(dump_file)
                print 'Average cost in epoch %d: %f' % (epoch_idx, cost_acc /
                                                        iters_in_epoch)
                cost_acc = 0
                c, cmask, u, umask, a, evidence = next(train_batch)
                train_pred = self.model.pred(c, cmask, u, umask)
                train_acc = np.mean(train_pred == a)
                #train_acc /= iters_in_epoch
                c, cmask, u, umask, a, evidence = next(test_batch)
                if self.oo['dropout']:
                    self.model.use_noise.set_value(0)
                test_pred = self.model.pred(c, cmask, u, umask)
                if self.oo['dropout']:
                    self.model.use_noise.set_value(1)
                #test_acc_old = test_acc
                test_acc = np.mean(test_pred == a)
                test_acc_history.append(test_acc)
                print 'training accuracy: %f\ttest accuracy: %f' % (train_acc,
                                                                    test_acc)
                if 'linear_start' in self.oo.keys() and self.oo['linear_start'] and self.model.linear.get_value() == 1:
                    if epoch_idx > self.oo['linear_start_lazy'] and test_acc <= np.min(test_acc_history[-(self.oo['linear_start_lazy']+1):-1]):
                        print '-' * 8 + "End linear start" + '-' * 8
                        self.model.linear.set_value(0.)

                if 0 < epoch_idx <= 100 and epoch_idx % self.oo['decay_period'] == 0:
                    lr *= self.oo['decay']
                    print "lr decays to %f" % lr
                train_acc = 0
                epoch_idx += 1
                if epoch_idx >= max_epoch:
                    break
        self.model.dump_params(dump_file)

def preprocess_options(options, disp=False):
    if disp:
        print "options:\n", json.dumps(options, indent=4, sort_keys=False)

    log_options = options['log_options']
    if log_options['dump_config']:
        path = Path(log_options['dump_path'])
        if not path.exists():
            path.mkdir()
        dumpname = log_options['dump_name']
        basename = os.path.splitext(dumpname)[0] + '.json'
        json.dump(options,
                  open(
                      str(path / basename), 'w'),
                  indent=4,
                  sort_keys=False)

    data_readers = {'QAReader': QAReader,
                    'minibatch': MinibatchReader}

    options['data_options']['reader'] \
    = data_readers[options['data_options']['reader']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--options',
        type=str, default="configs/memn2n.json")
    parser.add_argument(
        '-t', '--task',
        type=int, default=0)
    parser.add_argument('-do',
                        '--display_options',
                        dest='display_options',
                        action='store_true')
    parser.add_argument('-ndo',
                        '--no_display_options',
                        dest='display_options',
                        action='store_false')
    parser.set_defaults(display_options=True)
    args = parser.parse_args()
    options = json.load(open(args.options, 'r'))
    if args.task > 0:
        options['data_options']['task_number'] = args.task
        options['optimization_options']['dump_name'] = \
            options["model_options"]["model_name"] + "_%d.pkl" % args.task
    preprocess_options(options, args.display_options)

    exp = QATask(options)
    exp.build_model()
    exp.train()
