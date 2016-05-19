import QAReader
from experiment import experiment

base_path = 'bAbI/en-10k/'
train_path = base_path + 'qa10_indefinite-knowledge_train.txt'
test_path = base_path + 'qa10_indefinite-knowledge_test.txt'
data_options = {
    'train_path': test_path, # Train on a smaller dataset, to make it overfit
    'test_path': test_path,
    'reader': QAReader.QAReader,
    'word_to_index': None, ####################
    }
optimization_options = {
    'learning_rate': 0.01,
    'batch_size_train': 5,
    'batch_size_test': 3,
    'optimizer': 'rmsprop',
    'max_epoch': 30,
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

exp = experiment(options)
exp.build_model()
exp.train()
