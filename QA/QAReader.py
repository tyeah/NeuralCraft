import re
from collections import Counter


def get_or_return_UNKNOWN(dictionary, key):
    return dictionary.get(key, dictionary['<UNKNOWN>'])


class QAReader:
    """QAReader: reader for bAbI qa files
    Each file contains sentences in either of the following two patterns
    > ID text
    > ID question[tab]answer[tab]supporting fact IDS.
    The IDs of a story appears in 1-based increasing order. When

    The reader reads each sentence and parse them into:
          Story
              |- Context
              |- Question

    The reader also indexes the word, producing
    > index_to_word: a list of words
    > word_to_index: a dict mapping words to index
    The words are indexed in alphabetical order, also for generality, a threshold parameter will exclude any words that appear too infrequently.

    The sentences are segmented into words by a default segmenter. Right now it simply splits the words by spaces and split the punctuation (period or question mark) at the end. For more sophisticated sentences provide a better segmenter.

    Method:
    getDictionaries () => (dict, dict):
        Return (index_to_word, word_to_index) to initialize another QAReader
        with a preset dictionary instead of building them from scratch.

        e.g.
        qa1 = QAReader('file1.txt')
        qa2 = QAReader('file2.txt', dictionaries=qa1.getDictionaries())
    """

    def __init__(self,
                 filename,
                 threshold=3,
                 segmenter=None,
                 dictionaries=None,
                 include_testset_vocab=True):
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
                id, string = splitter.match(line).groups()
                id = int(id)
                string = string.lower()
                if id == 1:
                    if story is not None:
                        self.stories.append(story)
                    story = Story()
                    context_id_mapping = {}
                    true_index = 0
                    index_id_mapping = {}

                if '?' in string:  # this is a question
                    question, answer, evidences = string.split('\t')
                    question = segmenter(question[:-1])
                    word_counter.update(question)
                    word_counter[answer] += 1
                    quest = Question(question, answer, self.word_to_index)
                    for evid in map(int, evidences.split()):
                        quest.evidence_indexes.append(index_id_mapping[evid])
                        quest.evidences.append(context_id_mapping[evid])
                    story.questions.append(quest)
                else:
                    context = segmenter(string[:-1])
                    word_counter.update(context)
                    con = Context(context, self.word_to_index)
                    context_id_mapping[id] = con
                    index_id_mapping[id] = true_index
                    true_index += 1
                    story.contexts.append(con)
            if story is not None:  # last story
                self.stories.append(story)

            if include_testset_vocab:
                all_words = sorted(word_counter.keys())
                for word in all_words:
                    if word_counter[
                            word] >= threshold and word not in self.word_to_index:
                        self.word_to_index[word] = len(self.index_to_word)
                        self.index_to_word.append(word)

            self.specialWords = {w: self.word_to_index[w] for w in tokens}

    @staticmethod
    def segment(sent):
        words = sent.split()
        if not words[-1].isalpha():
            lastword = words[-1]
            words[-1] = lastword.strip(',.!?')
            words.append(lastword[-1])
        return words

    def getDictionaries(self):
        return self.index_to_word, self.word_to_index

    def __repr__(self):
        return '< Q&A Reader with {} stories. Vocab size = {} >'.format(
            len(self.stories), len(self.index_to_word))


class Context:
    def __init__(self, string, dictionary):
        self.string = string
        self.dictionary = dictionary

    def toIndex(self):
        return [get_or_return_UNKNOWN(self.dictionary, w) for w in self.string]

    def __repr__(self):
        return ' '.join(self.string)


class Question:
    def __init__(self, question, answer, dictionary):
        self.question = question
        self.answer = answer
        self.evidences = []
        self.evidence_indexes = []
        self.dictionary = dictionary

    def toIndex(self):
        return {'question': [get_or_return_UNKNOWN(self.dictionary, w)
                             for w in self.question],
                'answer': get_or_return_UNKNOWN(self.dictionary, self.answer),
                'evidence_indices': self.evidence_indexes}

    def __repr__(self):
        return '<Question: {}\n Answer: {}\n Supporting fact: {}>'.format(
            ' '.join(self.question), self.answer, str(self.evidences))


class Story(object):
    """docstring for Story"""

    def __init__(self):
        self.contexts = []
        self.questions = []

    def __repr__(self):
        return '< Q&A Story with {} context sentences, {} questions >'.format(
            len(self.contexts), len(self.questions))
