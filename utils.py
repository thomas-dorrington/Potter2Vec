import nltk
import regex
import random
import numpy as np
from progress.bar import Bar
from numpy.linalg import norm


class DocumentNotInCollection(Exception):
    """
    Raised if we try and get a document vector from a (possible weighted) term-document matrix
    that never existed in the training corpus of documents.
    """
    pass


class WordNotInVocabulary(Exception):
    """
    Raised if we try and get a word vector from a (possibly weighted) co-occurrence matrix,
    either because the word does not occur in the training set at all,
    or occurs with a frequency less than the minimum allowable threshold the VSM was trained on.
    """
    pass


class MySentences(object):
    """
    Class to iterate over the sentence and tokens in a series of text files, generator-style.
    Saves us from loading everything into memory at once during the training of vector space models.
    """

    def __init__(self, files):
        """
        `files` is a list of paths to the text files.
        For the Harry Potter series, this will be a list of length 7, but easily applicable to any list of text files.
        """

        self.files = files  # List of (relative) paths to text files
        self.iteration = 0   # How many times we have iterated through all the text files

    def __iter__(self):
        """
        Implement iteration over class generator-style
        """

        bar = Bar("Processing", max=len(self.files))
        print("Starting iteration %s" % str(self.iteration))

        # Shuffle order in which we present text files to improve training
        # Really ought to shuffle order sentences are presented, but this makes iterating like a generator harder
        # Could maybe generate 20 at a time, shuffle, then yield.
        random.shuffle(self.files)

        for f in self.files:

            with open(f, 'r') as open_f:

                for line in open_f:

                    line = line.strip()

                    if line:
                        # If line is not just whitespace

                        sents = nltk.sent_tokenize(line.decode('utf8'))  # Line in text      -> list of sentences
                        sents = [nltk.word_tokenize(s) for s in sents]   # List of sentences -> list of list of tokens
                        sents = [self.preprocess(s) for s in sents]      # Pre-process each token in each sent

                        for s in sents:
                            if s:
                                # If non-empty list of tokens
                                yield s

            bar.next()

        bar.finish()

        self.iteration += 1

    def preprocess(self, tokens):
        """
        Clean line of text before yielding for VSM training.
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
                # If there are some characters left
                return_tokens.append(tok)

        return return_tokens


class MyNGrams(object):
    """
    Class to iterate over sentences from a list of files,
    transforming sequence of commonly occurring tokens into n-grams using `gensim.models.phrases` package.
    For example, when iterating over text files, rather than yielding
     ["Harry", "Potter", "cast", "Expecto", "Patronum"],
    we might yield
     ["Harry_Potter", "cast", "Expecto_Patronum"]
    We are treating these commonly occurring n-grams as individual tokens, each with their own vector embedding.
    """

    def __init__(self, files, n_gram_phraser):
        """
        `n_gram_phraser` is a `gensim.models.phrases.Phraser` object.
        While a `Phrases` object would also work in this setting, a Phraser is much more efficient, time and space wise.
        """

        self.n_gram_phraser = n_gram_phraser
        self.sentences = MySentences(files=files)

    def __iter__(self):

        for s in self.sentences:
            yield self.n_gram_phraser[s]


class Vocabulary(object):
    """
    Class to generate and store a vocabulary over a training document corpus. Just a wrapper for a vocab dictionary.
    Stores a unique integer with each vocabulary word, making it easy to generate one-hot vectors,
    or use row/column indices in a word-word matrix or term-document matrices.
    """

    def __init__(self, files, min_count=1):
        """
        `files` is a list of (relative) paths to text files that constitute our training document collection.

        `min_count` is the number of times a word must occur for it to be considered an entry in the vocab.
        """

        self.min_count = min_count
        self.vocab = self._build_vocab(files=files)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item):
        return self.vocab[item]

    def __iter__(self):
        for k in sorted(self.vocab, key=self.vocab.get):
            # Sort on values (i.e. unique increasing integers)
            yield k

    def _build_vocab(self, files):
        """
        Build a vocabulary, i.e. a mapping from (unique) word type to (unique) integer.
        """

        sentences = MySentences(files=files)

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


def cosine_similarity(vector1, vector2):
    """
    Returns (one of the better) measures of similarity between two vectors: the cosine of the angle between them.
    Both `vector1` and `vector2` are just a list of numbers.
    """

    return (np.dot(vector1, vector2))/(norm(vector1) * norm(vector2))


def most_similar(matrix, word, top_n=5, similarity_measure=cosine_similarity):
    """
    Returns the `top_n` most similar words to `word` under the vector embedding model defined by the `matrix` object.
    `matrix` has an attribute, also called `matrix`, that is the actual matrix of numbers.
    `similarity_measure` is the way we are defining similarity, typically cosine similarity.
    """

    # The `vocab` attribute of `matrix` is a map from word to unique integer, corresponding to the row index in matrix
    vocab = matrix.vocab

    if word not in vocab:
        raise WordNotInVocabulary(word)

    vector = matrix.matrix[vocab[word]]

    # `best_seen` is a list of tuples of form: (some word, similarity of it to `word`)
    # We keep this list sorted at all times, so to insert a new word is very simple:
    # Check if new word has a similarity greater than the word with the smallest index; if so, replace, and re-sort.
    # There is a time-complexity trade off here: sorting at the end of each insertion has a cost,
    # but it means we don't need to iterate over the list every time to find the smallest element.
    # (Also makes the code much clearer)
    best_seen = []

    for other_word in vocab:

        if other_word == word:
            # A word and itself have a similarity of 1.0; not very useful
            continue

        other_vector = matrix.matrix[vocab[other_word]]
        similarity = similarity_measure(vector, other_vector)

        if len(best_seen) < top_n:
            # Haven't found `top_n` elements yet; keep adding
            best_seen.append((other_word, similarity))
        else:
            if similarity > best_seen[0][1]:
                # Replace element with smallest similarity
                best_seen[0] = (other_word, similarity)

        # Re-sort before next-iteration
        best_seen = sorted(best_seen, key=lambda x: x[1])

    # Return in descending order of similarity for clarity
    return sorted(best_seen, key=lambda x: x[1], reverse=True)
