import os
import argparse
from pathlib2 import Path
import json
from time import time

import numpy as np

from QAReader import QAReader
import Models

def read_dataset(data_path, task_num, lang, Reader):
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
        train_set = Reader(train_fn)
        test_set = Reader(test_fn,
                        dictionaries=train_set.getDictionaries())
        # This below recreates the situation of experiment.py
        # test_set = Reader(train_fn, dictionaries=train_set.getDictionaries())
    else:
        raise ValueError('Invalid task number: %d' % task_num)
    return train_set, test_set


class QATask(object):
    def __init__(self, options):
        self.do = options['data_options']
        data_path = self.do['data_path']
        task_num = self.do['task_number']
        lang = self.do.get('language', 'en')  # defaults to use small Eng set
        self.qa_train, self.qa_test \
            = read_dataset(data_path,
                           task_num, lang, options['data_options']['reader'])

        self.data_size = len(self.qa_train.stories)

        tokens = self.qa_train.specialWords
        self.NULL = tokens['<NULL>']
        self.EOS = tokens['<EOS>']
        self.UNKNOWN = tokens['<UNKNOWN>']

        self.mo = options['model_options']
        self.oo = options['optimization_options']
        self.lo = options['log_options']

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
        train_batch = self.minibatch(self.qa_train,
                                     self.oo['batch_size_train'], True)
        test_batch = self.minibatch(self.qa_test, self.oo['batch_size_test'],
                                    True)
        print 'Starting training...'
        epoch_idx = 0
        iter_idx = 0
        cost_acc = 0
        while (True):
            c, cmask, u, umask, a = next(train_batch)
            cost = self.model.update(c, cmask, u, umask, a, lr)
            cost_acc += cost
            if iter_idx % disp_iter == 0:
                print 'cost at epoch %d, iteration %d: %f' % (epoch_idx,
                                                              iter_idx, cost)
            iter_idx += 1
            dump_file = os.path.join(self.oo['weight_path'],
                                     self.oo['dump_name'])
            if iter_idx % iters_in_epoch == 0:
                if epoch_idx > 0 and epoch_idx % self.lo["dump_epoch"] == 0:
                    self.model.dump_params(dump_file)
                lr *= self.oo['decay']
                print 'Average cost in epoch %d: %f' % (epoch_idx, cost_acc /
                                                        iters_in_epoch)
                epoch_idx += 1
                cost_acc = 0
                c, cmask, u, umask, a = next(train_batch)
                train_pred = self.model.pred(c, cmask, u, umask)
                train_acc = np.mean(train_pred == a)
                c, cmask, u, umask, a = next(test_batch)
                test_pred = self.model.pred(c, cmask, u, umask)
                test_acc = np.mean(test_pred == a)
                print 'training accuracy: %f\ttest accuracy: %f' % (train_acc,
                                                                    test_acc)
                if epoch_idx >= max_epoch:
                    self.model.dump_params(dump_file)
                    break

    def minibatch(self, qa, batch_size=None, shuffle=True):
        data_size = len(qa.stories)
        data_idx = range(data_size)
        if batch_size == None or batch_size > data_size:
            batch_size = data_size
        sen_len = self.mo['sentence_length']
        ctx_len = self.mo['context_length']
        start, end = 0, 0

        def sentence_process(sentence):
            s = np.ones(sen_len) * self.NULL
            m = np.ones(sen_len) * self.NULL
            m[:len(sentence)] = 1
            if len(sentence) >= sen_len:
                s[:] = sentence[:sen_len]
            else:
                s[:len(sentence)] = sentence
                s[len(sentence)] = self.EOS
            return s, m

        def context_process(context):
            c = np.ones((ctx_len, sen_len)) * self.NULL
            m = np.zeros((ctx_len, sen_len))
            context = context[:ctx_len]
            cm = np.array([sentence_process(s) for s in context])
            c[:len(context), :] = cm[:, 0]
            m[:len(context), :] = cm[:, 1]
            return c, m

        while (True):
            end = start + batch_size
            if end >= data_size:
                end %= data_size
                batch_idx = data_idx[start:data_size]
                if (shuffle):
                    np.random.shuffle(data_idx)
                batch_idx.extend(data_idx[0:end])
            else:
                batch_idx = data_idx[start:end]
            start = end
            stories = [qa.stories[idx] for idx in batch_idx]
            questions = [np.random.choice(st.questions) for st in stories]
            ret_c = np.array([context_process([c.toIndex() for c in st.contexts
                                               ]) for st in stories])
            ret_q = np.array([sentence_process(q.toIndex()['question'])
                              for q in questions])
            ret_a = np.array([q.toIndex()['answer'] for q in questions])
            yield (ret_c[:, 0], ret_c[:, 1], ret_q[:, 0], ret_q[:, 1], ret_a)


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

    data_readers = {'QAReader': QAReader}

    options['data_options']['reader'] \
    = data_readers[options['data_options']['reader']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--options',
        type=str, default="configs/memn2n.json")
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
    preprocess_options(options, args.display_options)

    exp = QATask(options)
    exp.build_model()
    exp.train()
