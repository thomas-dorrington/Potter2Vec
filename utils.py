import nltk
import regex
import random
from progress.bar import Bar


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


class Vocabulary(object):
    """
    Class to store a vocabulary over a piece of text. Just a wrapper for a vocab dictionary.
    Initialized with a generator over all sentences we want to build the vocabulary over.
    Stores a unique integer with each vocabulary word, making it easy to generate one-hot vectors,
    or use row/column indices in a word-word matrix or term-document matrix.
    """

    def __init__(self, sentences, min_count=1):
        """
        `sentences` is an iterator over all the pre-processed tokens & sentences in the text to build vocab over.
        `min_count` is the number of times a word must occur for it to be considered an entry in the vocab.
        """

        self.min_count = min_count
        self.vocab = self._build_vocab(sentences=sentences)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item):
        return self.vocab[item]

    def __iter__(self):
        for k in sorted(self.vocab, key=self.vocab.get):
            # Sort on values (i.e. unique increasing integers)
            yield k

    def _build_vocab(self, sentences):
        """
        Build a vocabulary, i.e. a mapping from (unique) word type to (unique) integer.
        """

        # Initialize a dictionary, mapping from word to its count in training corpus.
        # The count is used to prune elements not above a certain minimum count,
        # but is replaced with a unique integer instead after counting before returning.
        vocab = {}

        for sent in sentences:
            for tok in sent:
                vocab[tok] = 1 if tok not in vocab else vocab[tok] + 1

        # Prune any vocab elements not above the minimum count and assign tokens a unique ascending integer
        i = 0
        new_vocab = {}

        for tok, count in vocab.iteritems():
            if count >= self.min_count:
                new_vocab[tok] = i
                i += 1

        return new_vocab
