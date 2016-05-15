# get dataset
# minibatch
# training
# pass word_index to reader

import QAReader
import numpy as np
import Models
from time import time

base_path = 'bAbI/en-10k/'
train_path = base_path + 'qa10_indefinite-knowledge_train.txt'
test_path = base_path + 'qa10_indefinite-knowledge_test.txt'
data_options = {
    'train_path': train_path, 
    'test_path': test_path, 
    'reader': QAReader.QAReader,
    'word_to_index': None, ####################
    }
optimization_options = {
    'learning_rate': 0.01,
    'batch_size_train': 5,
    'batch_size_test': 3,
    'optimizer': 'rmsprop',
    'max_epoch': 10,
    'verbose': True,
    'disp_iter': 20,
    }
model_options = {
    'model_name': 'lstm',
    'context_length': 8,
    'sentence_length': 32,
    'embedding_size': 16,
    'num_hid': 32,
    'vocab_size': None, #####################
    }
options = {
    'data_options': data_options,
    'optimization_options': optimization_options,
    'model_options': model_options,
    }


class experiment(object):
  def __init__(self, options):
    self.do = options['data_options']
    qa = self.do['reader'](self.do['train_path'])
    vocab_size = len(qa.index_to_word)
    options['model_options']['vocab_size'] = vocab_size
    self.data_size = len(qa.stories)

    self.oo = options['optimization_options']
    self.mo = options['model_options']
    self.batch_size_train = self.oo['batch_size_train']
    self.batch_size_test = self.oo['batch_size_test']

    self.verbose = self.oo['verbose']

    self.model = Models.model(options)

  def verbose_print(self, s):
    if self.verbose:
      print(s)

  def build_model(self):
    self.verbose_print('Starting compiling...')
    t0 = time()
    self.model.build()
    self.verbose_print('Ending compiling...\nCompiling time: %.2f' % (time() - t0))

  def train(self):
    max_epoch = self.oo['max_epoch']
    iters_in_epoch = int(self.data_size / self.oo['batch_size_train'])
    lr = self.oo['learning_rate']
    disp_iter = self.oo['disp_iter']
    train_batch = self.minibatch(self.do['train_path'], self.oo['batch_size_train'], True)
    print 'Starting training...'
    epoch_idx = 0
    iter_idx = 0
    cost_acc = 0
    while(True):
      c, cmask, u, umask, a = next(train_batch)
      cost = self.model.update(c, cmask, u, umask, a, lr)
      cost_acc += cost
      if iter_idx % disp_iter == 0:
        print 'cost at epoch %d, iteration %d: %f' % (iter_idx, epoch_idx, cost)
      iter_idx += 1
      if iter_idx % iters_in_epoch == 0:
        print 'Average cost in epoch %d: %f' % (epoch_idx, cost_acc / iters_in_epoch)
        epoch_idx += 1
        cost_acc = 0
        if epoch_idx >= max_epoch:
          break


  def minibatch(self, file_path, batch_size=None, shuffle=True):
    qa = self.do['reader'](file_path)
    data_size = len(qa.stories)
    data_idx = range(data_size)
    if batch_size == None or batch_size > data_size:
      batch_size = data_size
    sen_len = self.mo['sentence_length']
    ctx_len = self.mo['context_length']
    start, end = 0, 0

    def sentence_process(sentence):
      s = np.zeros(sen_len)
      m = np.zeros(sen_len)
      m[:len(sentence)] = 1
      if len(sentence) > sen_len:
        s[:] = sentence[:sen_len]
      else:
        s[:len(sentence)] = sentence
      return s, m

    def context_process(context):
      c = np.zeros((ctx_len, sen_len))
      m = np.zeros((ctx_len, sen_len))
      context = context[:ctx_len]
      cm = np.array([sentence_process(s) for s in context])
      c[:len(context), :] = cm[:, 0]
      m[:len(context), :] = cm[:, 1]
      return c, m

    while(True):
      end = start + batch_size
      if end >= data_size:
        end %= data_size
        batch_idx = data_idx[start : data_size]
        if (shuffle):
          np.random.shuffle(data_idx)
        batch_idx.extend(data_idx[0 : end])
      else:
        batch_idx= data_idx[start : end]
      start = end
      stories = [qa.stories[idx] for idx in batch_idx]
      questions = [np.random.choice(st.questions) for st in stories]
      ret_c = np.array([context_process([c.toIndex() for c in st.contexts]) for st in stories])
      ret_q = np.array([sentence_process(q.toIndex()['question']) for q in questions])
      ret_a = np.array([q.toIndex()['answer'] for q in questions])
      yield (ret_c[:, 0], ret_c[:, 1], ret_q[:, 0], ret_q[:, 1], ret_a)


def main():
  exp = experiment(options)
  exp.build_model()
  exp.train()

main()
