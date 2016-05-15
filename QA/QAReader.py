import re
from collections import Counter

class Context:
    def __init__(self, string, dictionary):
        self.string = string
        self.dictionary = dictionary

    def toIndex(self):
        return [self.dictionary[w] for w in self.string]

    def __repr__(self):
        return ' '.join(self.string)

class Question:
    def __init__(self, question, answer, dictionary):
        self.question = question
        self.answer = answer
        self.evidences = []
        self.dictionary = dictionary

    def toIndex(self):
        return {'question': [self.dictionary[w] for w in self.question],
                'answer': self.dictionary[self.answer] }

    def __repr__(self):
        return '<Question: {}\n Answer: {}\n Supporting fact: {}>'.format(' '.join(self.question), self.answer, str(self.evidences))

class Story(object):
    """docstring for Story"""
    def __init__(self):
        self.contexts = []
        self.questions = []

    def __repr__(self):
        return '< Q&A Story with {} context sentences, {} questions >'.format(len(self.contexts), len(self.questions))

class QAReader:
    """QAReader: reader for bAbI qa files
    Each file contains sentences in either of the following two patterns
    > ID text
    > ID question[tab]answer[tab]supporting fact IDS.
    The IDs of a story appears in 1-based increasing order. When

    The reader reads each sentence and parse them into:
          Story
              |- Context
              |- Questoin

    The reader also indexes the word, producing
    > index_to_word: a list of words
    > word_to_index: a dict mapping words to index
    The words are indexed in alphabetical order, also for generality, a threshold parameter will exclude any words that appear too infrequently.

    The sentences are segmented into words by a default segmenter. Right now it simply splits the words by spaces and split the punctuation (period or question mark) at the end. For more sophisticated sentences provide a better segmenter.
    """
    def __init__(self, filename,
                 threshold=3,
                 segmenter=None):
        if not segmenter:
            segmenter = self.segment

        splitter = re.compile('(\d+) (.+)')
        story = None

        word_counter = Counter()
        self.index_to_word = ['NULL', 'EOS']
        self.word_to_index = {'NULL': 0, 'EOS': 1}
        with open(filename, 'r') as file:
            self.stories = []
            for line in file:
                id, string = splitter.match(line).groups()
                id = int(id)
                if id == 1:
                    if story is not None:
                        self.stories.append(story)
                    story = Story()
                    context_id_mapping = {}

                if '?' in string: # this is a question
                    question, answer, evidences = string.split('\t')
                    question = segmenter(question)
                    word_counter.update(question)
                    word_counter[answer] += 1
                    quest = Question(question, answer, self.word_to_index)
                    for evid in map(int, evidences.split()):
                        quest.evidences.append(context_id_mapping[evid])
                    story.questions.append(quest)
                else:
                    context = segmenter(string)
                    word_counter.update(context)
                    con = Context(context, self.word_to_index)
                    context_id_mapping[id] = con
                    story.contexts.append(con)
            if story is not None: # last story
                self.stories.append(story)

            all_words = sorted(word_counter.keys())
            for word in all_words:
                if word_counter[word] >= threshold:
                    self.word_to_index[word] = len(self.index_to_word)
                    self.index_to_word.append(word)
            self.word_to_index['<UNKNOWN>'] = len(self.index_to_word)
            self.index_to_word.append('<UNKNOWN>')

    @staticmethod
    def segment(sent):
        words = sent.split()
        if not words[-1].isalpha():
            lastword = words[-1]
            words[-1] = lastword.strip(',.!?')
            words.append(lastword[-1])
        return words
