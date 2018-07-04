import os
import nltk
import regex
import random
from progress.bar import Bar
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases


class MySentences(object):
    """
    Class to iterate over the sentence and tokens in the Harry Potter series.
    Saves us from loading everything into memory at once.
    """

    def __init__(self, txt_files, verbose=True):
        """
        `txt_files` is a list of paths to the txt files (relative wrt. this script).
        For the Harry Potter, this will be a list of length 7, but easily applicable to any list of text files.
        """

        self.iteration = 0
        self.verbose = verbose
        self.txt_files = txt_files

    def __iter__(self):
        """
        `Implement iteration over class as a generator
        """

        if self.verbose:
            bar = Bar("Processing", max=len(self.txt_files))
            print("Starting iteration %s" % str(self.iteration))

        # Shuffle order in which we present txt files to improve training
        random.shuffle(self.txt_files)

        for f in self.txt_files:

            with open(f, 'r') as open_f:

                for line in open_f:

                    line = line.strip()

                    if line:

                        sents = nltk.sent_tokenize(line.decode('utf8'))

                        sents = [nltk.word_tokenize(s) for s in sents]
                        sents = [MySentences.preprocess(s) for s in sents]

                        for s in sents:
                            if s:
                                yield s

            if self.verbose:
                bar.next()

        if self.verbose:
            bar.finish()

        self.iteration += 1

    @staticmethod
    def preprocess(tokens):
        """
        Clean line of text before yielding for word2vec training.
        Takes a list of tokens `tokens`, and returns a pre-processed version of this list.

        For increased speed, rather than sentence segmenting, word tokenizing, and pre-processing each iteration,
        these pre-processed lists could be saved off to disk, so iteration consists solely of loading files & yielding.
        """

        return_tokens = []

        for tok in tokens:

            tok = tok.lower()
            tok = regex.sub(r'[^\p{Alpha} ]', '', tok, regex.U)  # Remove any character not alphabetical or a space
            tok = tok.strip()

            if tok:
                # If there are some characters left to add
                return_tokens.append(tok)

        return return_tokens


class MyNGrams(object):

    def __init__(self, txt_files, n_gram_phraser):

        self.n_gram_phraser = n_gram_phraser
        self.sentences = MySentences(txt_files=txt_files)

    def __iter__(self):

        for s in self.sentences:
            yield self.n_gram_phraser[s]


if __name__ == '__main__':

    n_iter = 20
    vector_size = 100

    potter_files = [os.path.join('data/', x) for x in os.listdir('data/')]

    potter_sents = MySentences(
        txt_files=potter_files
    )

    potter_phrases = Phrases(
        sentences=potter_sents,
        min_count=5,
        threshold=10.0
    )

    potter_phraser = Phraser(potter_phrases)

    potter_n_grams = MyNGrams(
        txt_files=potter_files,
        n_gram_phraser=potter_phraser
    )

    potter_word2vec = Word2Vec(
        sentences=potter_n_grams,  # Class to iterate over sentences, embedded using n-gram phraser
        size=vector_size,          # Dimension size of resulting vector embeddings
        window=5,                  # How many words to look either side of context word
        min_count=5,               # Minimum frequency of words before discarding from vocab
        workers=4,                 # How many threads to spawn - speeds up training, not 1st iteration to build vocab
        iter=n_iter                # We actually perform `iter`+1 iterations; the first is to build vocab
    )

    if not os.path.exists('models/'):
        os.mkdir('models/')

    potter_phrases.save('models/potter_phrases.bin')
    potter_phraser.save('models/potter_phraser.bin')
    potter_word2vec.save('models/potter_%s.bin' % str(vector_size))
