''' a MinibatchReader works like a minibatch function. After reading the document it returns a function to be called over and over '''

# import re
# from collections import Counter
import numpy as np

from QAReader import *

class MinibatchReader(QAReader):
    def __init__(self, filename, context_length, sentence_length, segmenter=None, dictionaries=None, **kws):
        if not segmenter:
            segmenter = self.segment

        splitter = re.compile('(\d+) (.+)')
        story = None

        self.sentence_length = sentence_length
        self.context_length = context_length

        word_counter = Counter()
        build_dictionary = dictionaries is None

        tokens = ['<NULL>', '<EOS>', '<UNKNOWN>']

        self.index_to_word = tokens if build_dictionary else dictionaries[0]
        self.word_to_index = {w: i for i, w in enumerate(tokens)} \
                            if build_dictionary else dictionaries[1]
        with open(filename, 'r') as file:
            self.stories = []
            for line in file:
                ''' foreach Sentence
                if is_Context: save for later
                if is question:
                    construct a story with just this question
                '''
                id, string = splitter.match(line).groups()
                id = int(id)
                if id == 1:
                    id_without_question = 1
                    context_id_mapping = {}
                    index_without_question_mapping = {}
                    contexts_list = []

                if '?' in string: # this is a question
                    question, answer, evidences = string.split('\t')
                    question = segmenter(question[:-1])
                    word_counter.update(question)
                    word_counter[answer] += 1
                    quest = Question(question, answer, self.word_to_index)
                    evs =  map(int, evidences.split())
                    quest.evidences = [context_id_mapping[id] for id in evs]
                    quest.evidence_indexes = [index_without_question_mapping[id] for id in evs]

                    story = Story()
                    story.contexts = contexts_list[:context_length]
                    story.questions = [quest]
                    self.stories.append(story)
                else:
                    index_without_question_mapping[id] = id_without_question
                    id_without_question += 1
                    context = segmenter(string[:-1]) # without the period
                    word_counter.update(context)
                    con = Context(context, self.word_to_index)
                    context_id_mapping[id] = con
                    contexts_list.append(con)
            if story is not None: # last story
                self.stories.append(story)

            if build_dictionary:
                all_words = sorted(word_counter.keys())
                for word in all_words:
                    self.word_to_index[word] = len(self.index_to_word)
                    self.index_to_word.append(word)

            self.specialWords = {w: self.word_to_index[w] for w in tokens}
            self.NULL = self.specialWords['<NULL>']
            self.EOS = self.specialWords['<EOS>']
            self.UNKNOWN = self.specialWords['<UNKNOWN>']


    def minibatch(self, batch_size=None, shuffle=True):
        data_size = len(self.stories)
        data_idx = range(data_size)
        if batch_size == None or batch_size > data_size:
            batch_size = data_size
        sen_len = self.sentence_length
        ctx_len = self.context_length
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

        #q_idx = np.random.randint(4)
        while (True):
            end = start + batch_size
            if end >= data_size:
                end %= data_size
                batch_idx = data_idx[start:data_size]
                if (shuffle):
                    np.random.shuffle(data_idx)
                batch_idx.extend(data_idx[0:end])
                #q_idx = np.random.randint(4)
            else:
                batch_idx = data_idx[start:end]
            #print batch_idx
            start = end
            stories = [self.stories[idx] for idx in batch_idx]
            questions = [np.random.choice(st.questions) for st in stories]
            #q_idx = 0
            #questions = [st.questions[q_idx] for st in stories]
            ret_c = np.array([context_process([c.toIndex() for c in st.contexts
                                               ]) for st in stories])
            ret_q = np.array([sentence_process(q.toIndex()['question'])
                              for q in questions])
            ret_a = np.array([q.toIndex()['answer'] for q in questions])
            ret_e = [q.toIndex()['evidence_indices'] for q in questions]
            yield (ret_c[:, 0], ret_c[:, 1], ret_q[:, 0], ret_q[:, 1], ret_a, ret_e)

if __name__ == '__main__':
    # test
    ar = MinibatchReader('bAbI/en/qa1_single-supporting-fact_test.txt', 10, 32)
    # print "qa1 contains %d questions" % len(ar.stories)
    # for s in ar.stories[:10]:
    #     print "Question: ", s.questions[0]
    #     print "Contexts: "
    #     for c in s.contexts:
    #         print c
    #     print "Evidence indexes: ", s.questions[0].evidence_indexes
    #     print
    batch = ar.minibatch()
    print next(batch)
