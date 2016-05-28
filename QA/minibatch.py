''' a MinibatchReader works like a minibatch function. After reading the document it returns a function to be called over and over '''

# import re
# from collections import Counter

from QAReader import *

class ApproximateReader(QAReader):
    def __init__(self, filename, context_length, segmenter=None, dictionaries=None):
        if not segmenter:
            segmenter = self.segment

        splitter = re.compile('(\d+) (.+)')
        story = None

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

if __name__ == '__main__':
    # test
    ar = ApproximateReader('bAbI/en/qa1_single-supporting-fact_test.txt', 10)
    print "qa1 contains %d questions" % len(ar.stories)
    for s in ar.stories[:10]:
        print "Question: ", s.questions[0]
        print "Contexts: "
        for c in s.contexts:
            print c
        print "Evidence indexes: ", s.questions[0].evidence_indexes
        print
