from QAReader import *

class MonoSentReader(QAReader):
    def __init__(self, filename, segmenter=None, dictionaries=None):
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
                    split, find supporting fact
                    Raise error if more than one supporting fact.
                    construct a story with just this question and ONLY the supporting fact.
                '''
                id, string = splitter.match(line).groups()
                id = int(id)
                if id == 1:
                    context_id_mapping = {}
                if '?' in string: # this is a question
                    question, answer, evidences = string.split('\t')
                    question = segmenter(question[:-1])
                    word_counter.update(question)
                    word_counter[answer] += 1
                    quest = Question(question, answer, self.word_to_index)
                    evs =  map(int, evidences.split())
                    if len(evs) > 1:
                        raise ValueError('This reader requires all questions to have only a single supporting fact. Please do not use it along with qa2, 3, 7, 8, 11, 13-19')
                    evid = evs[0]
                    con = context_id_mapping[evid]

                    story = Story()
                    story.contexts.append(con)
                    quest.evidences.append(con)
                    story.questions.append(quest)
                    self.stories.append(story)
                else:
                    context = segmenter(string[:-1]) # without the period
                    word_counter.update(context)
                    con = Context(context, self.word_to_index)
                    context_id_mapping[id] = con
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
    msr = MonoSentReader('bAbI/en/qa1_single-supporting-fact_test.txt')
    print "qa1 contains %d questions" % len(msr.stories)
    for s in msr.stories[:10]:
        print "Question: ", s.questions[0]
