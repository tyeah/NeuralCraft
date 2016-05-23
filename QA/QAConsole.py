import os
from time import time
from collections import Counter
import argparse
import json

import numpy as np

from QAProblem import QATask, preprocess_options
from QAReader import QAReader, Story, Context, Question

class QAReaderMock(QAReader):
    def __init__(self, threshold=3, dictionaries=None):
        self.story = Story()
        self.stories = [ self.story ]

        self.word_counter = Counter()
        self.build_dictionary = dictionaries is None
        tokens = ['<NULL>', '<EOS>', '<UNKNOWN>']
        self.index_to_word = tokens if self.build_dictionary \
                                    else dictionaries[0]
        self.word_to_index = {w: i for i, w in enumerate(tokens)} \
                            if self.build_dictionary else dictionaries[1]

    def add_context(self, sent):
        words = self.segment(sent)

        if self.build_dictionary:
            self.word_counter.update(words)
            all_words = sorted(self.word_counter.keys())
            for word in all_words:
                self.word_to_index[word] = len(self.index_to_word)
                self.index_to_word.append(word)

        self.story.contexts.append( Context(words, self.word_to_index) )

    def add_question(self, sent, answer=None):
        words = self.segment(sent)
        self.story.questions = [ Question(words, answer, self.word_to_index) ]

class QAConsole(QATask):
    def __init__(self, options):
        super(QAConsole, self).__init__(options)
        self.options = options
        self.online_qa = QAReaderMock(dictionaries=self.qa_train.getDictionaries())

    def repl(self):
        while True:
            try:
                sent = raw_input('>> ')
            except EOFError:
                return

            if sent.endswith('.'):
                self.add_context(sent)
            elif sent.endswith('?'):
                print self.answer_question(sent)
            elif sent == 'reset':
                self.reset_context()

    def add_context(self, sent):
        self.online_qa.add_context(sent)

    def answer_question(self, sent):
        self.online_qa.add_question(sent)
        batch = self.minibatch(self.online_qa,
                               self.oo['batch_size_test'],
                               True)
        c, cmask, u, umask, a = next(batch)
        test_pred = self.model.pred(c, cmask, u, umask)
        return self.interprete(test_pred)

    def interprete(self, embedding):
        index = embedding[0]
        return self.online_qa.index_to_word[index]

    def reset_context(self):
        self.online_qa = QAReaderMock(self.options)

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
    parser.set_defaults(display_options=False)
    args = parser.parse_args()
    options = json.load(open(args.options, 'r'))
    preprocess_options(options, args.display_options)

    con = QAConsole(options)
    con.build_model()
    # con.train()
    con.repl()
