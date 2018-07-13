import os
import nltk
import regex
import random
import argparse
from progress.bar import Bar
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases


class MySentences(object):
    """
    Class to iterate over the sentence and tokens in the Harry Potter series, generator-style.
    Saves us from loading everything into memory at once.
    """

    def __init__(self, txt_files, pos_tag=False, verbose=True):
        """
        `txt_files` is a list of paths to the txt files (relative wrt. this script).
        For the Harry Potter, this will be a list of length 7, but easily applicable to any list of text files.

        If `pos_tag` is set to True, rather than just return a list of words,
        we return a list of words paired with their associated Part of Speech tag

        If `verbose` set, we output various pieces of logging information throughuout the training procedure
        """

        self.iteration = 0
        self.verbose = verbose
        self.pos_tag = pos_tag
        self.txt_files = txt_files

    def __iter__(self):
        """
        `Implement iteration over class as a generator
        """

        if self.verbose:
            bar = Bar("Processing", max=len(self.txt_files))
            print("Starting iteration %s" % str(self.iteration))

        # Shuffle order in which we present txt files to improve training
        # Really ought to shuffle order sentences are presented, but this makes iteration like a generator harder
        random.shuffle(self.txt_files)

        for f in self.txt_files:

            with open(f, 'r') as open_f:

                for line in open_f:

                    line = line.strip()

                    if line:

                        sents = nltk.sent_tokenize(line.decode('utf8'))  # Line in text to list of sents
                        sents = [nltk.word_tokenize(s) for s in sents]   # List of sents to list of list of tokens
                        sents = [self.preprocess(s) for s in sents]      # Pre-process each token in each sent

                        for s in sents:
                            if s:
                                yield s

            if self.verbose:
                bar.next()

        if self.verbose:
            bar.finish()

        self.iteration += 1

    def preprocess(self, tokens):
        """
        Clean line of text before yielding for word2vec training.
        Takes a list of tokens `tokens`, and returns a pre-processed version of this list.

        For increased speed, rather than sentence segmenting, word tokenizing, and pre-processing each iteration,
        these pre-processed lists could be saved off to disk, so iteration consists solely of loading files & yielding.
        This becomes more crucial as we do POS tagging or Word Sense Disambiguation.
        """

        return_tokens = []

        if self.pos_tag:

            tokens = nltk.pos_tag(tokens)  # Returns a list of tuples: (token, POS tag)

            for tok, tag in tokens:

                tok = tok.lower()
                tok = regex.sub(r'[^\p{Alpha} ]', '', tok, regex.U)  # Remove any character not alphabetical or a space
                tok = tok.strip()

                if tok:
                    # If some characters left
                    return_tokens.append("%s_%s" % (tok, tag))

        else:

            for tok in tokens:

                tok = tok.lower()
                tok = regex.sub(r'[^\p{Alpha} ]', '', tok, regex.U)  # Remove any character not alphabetical or a space
                tok = tok.strip()

                if tok:
                    # If there are some characters left
                    return_tokens.append(tok)

        return return_tokens


class MyNGrams(object):
    """
    Class to iterate over sentences from a list of files,
    transforming sequence of commonly occurring tokens into n-grams using `gensim.models.phrases` package.
    """

    def __init__(self, txt_files, n_gram_phraser):

        self.n_gram_phraser = n_gram_phraser
        self.sentences = MySentences(txt_files=txt_files)

    def __iter__(self):

        for s in self.sentences:
            yield self.n_gram_phraser[s]


def train_simple_model(potter_files, dir_to_save, n_iter=20, vector_size=100, verbose=True):

    potter_sents = MySentences(txt_files=potter_files,
                               verbose=verbose)

    potter_word2vec = Word2Vec(
        sentences=potter_sents,
        size=vector_size,
        window=5,
        min_count=5,
        workers=4,
        iter=n_iter
    )

    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)

    potter_word2vec.save(os.path.join(dir_to_save, 'potter_simple_w2v_%s.bin' % str(vector_size)))


def train_phrases_model(potter_files, dir_to_save, n_iter=20, vector_size=100, verbose=True):
    """
    Method to train a word2vec model, using `gensim.models.phrases` to learn vectors for commonly occuring phrases too.
    For example, rather than learn separate vectors for "whomping" and "willow",
    we learn a vector for the combined "whomping_willow",
    whose meaning wil be more representative than the sum of its constituents.
    As a real life example, consider: new + york =/= new_york
    """

    potter_sents = MySentences(txt_files=potter_files,
                               verbose=verbose)

    potter_phrases = Phrases(sentences=potter_sents,
                             min_count=5,
                             threshold=10.0)

    potter_phraser = Phraser(potter_phrases)

    potter_n_grams = MyNGrams(txt_files=potter_files,
                              n_gram_phraser=potter_phraser)

    potter_word2vec = Word2Vec(
        sentences=potter_n_grams,
        size=vector_size,
        window=5,
        min_count=5,
        workers=4,
        iter=n_iter
    )

    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)

    potter_phrases.save(os.path.join(dir_to_save, 'potter_phrases.bin'))
    potter_phraser.save(os.path.join(dir_to_save, 'potter_phraser.bin'))
    potter_word2vec.save(os.path.join(dir_to_save, 'potter_phrases_w2v_%s.bin' % str(vector_size)))


def train_pos_model(potter_files, dir_to_save, n_iter=20, vector_size=100, verbose=True):

    potter_sents = MySentences(txt_files=potter_files,
                               verbose=verbose,
                               pos_tag=True)

    potter_word2vec = Word2Vec(
        sentences=potter_sents,
        size=vector_size,
        window=5,
        min_count=5,
        workers=4,
        iter=n_iter
    )

    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)

    potter_word2vec.save(os.path.join(dir_to_save, 'potter_pos_w2v_%s.bin' % str(vector_size)))


parser = argparse.ArgumentParser(description='Script for training a Word2Vec model over Harry Potter books.')
parser.add_argument('-m', '--mode', default='simple', type=str,
                    help='What way to train model: \'simple\', \'phrases\', \'pos\', \'word_sense\'')
parser.add_argument('-i', '--iter', default=20, type=int,
                    help='How many iterations to train model over.')
parser.add_argument('-s', '--size', default=100, type=int,
                    help='Size of resulting vector embeddings')
parser.add_argument('-d', '--dir', type=str, required=True,
                    help='Directory to save resulting models. If name clashes, will overwrite')
parser.add_argument('-v', '--verbose', type=bool, default=True,
                    help='Print useful information through process of training.')

if __name__ == '__main__':

    args = parser.parse_args()

    potter_files = [os.path.join('data/', x) for x in os.listdir('data/')]

    func_to_call = {
        'simple': train_simple_model,
        'phrases': train_phrases_model,
        'pos': train_pos_model
    }[args.mode]

    func_to_call(potter_files=potter_files,
                 dir_to_save=args.dir,
                 n_iter=args.iter,
                 vector_size=args.size,
                 verbose=args.verbose)
