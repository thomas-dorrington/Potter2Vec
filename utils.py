import os
import copy
import nltk
import regex
import random
import numpy as np
from progress.bar import Bar
from numpy.linalg import norm


class DocumentNotInCollection(Exception):
    """
    Raised if we try and get a document vector from a (weighted) term-document matrix
    that never existed in the training corpus of documents.
    """
    pass


class WordNotInVocabulary(Exception):
    """
    Raised if we try and get a word vector from a (weighted) co-occurrence matrix,
    either because the word does not occur in the training set at all,
    or occurs with a frequency less than the minimum allowable threshold the model was trained on.
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
        For the Harry Potter series, this will be a list of length 7, but easily applicable to any text files.
        """

        self.iteration = 0             # How many times we have iterated over all the text files - used for logging
        self.files = copy.copy(files)  # Copy so when we shuffle we don't affect any class `files` attribute

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

                        sents = nltk.sent_tokenize(line.decode('utf8'))  # Line in text      -> list of sentences
                        sents = [nltk.word_tokenize(s) for s in sents]   # List of sentences -> list of list of tokens
                        sents = [self.preprocess(s) for s in sents]      # Pre-process each token in each sent

                        for s in sents:
                            if s:
                                yield s

            bar.next()

        bar.finish()

        self.iteration += 1

    def preprocess(self, tokens):
        """
        Clean line of text before yielding for training.
        Takes a list of `tokens`, and returns a pre-processed version of this list.

        For increased speed, rather than sentence segmenting, word tokenizing, and pre-processing each iteration,
        these pre-processed lists could be saved off to disk, so iteration consists solely of loading files & yielding.
        """

        return_tokens = []

        for tok in tokens:

            tok = tok.lower()
            tok = regex.sub(r'[^\p{Alpha} ]', '', tok, regex.U)  # Remove any character not alphabetical or a space
            tok = tok.strip()                                    # Remove trailing or leading whitespace

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

    def __init__(self, files, n_gram_phrasers):
        """
        `n_gram_phrasers` is a list of `gensim.models.phrases.Phraser` objects.
        While a `Phrases` object would also work in this setting, a Phraser is much more efficient, time and space wise
        Each object incrementally joins collocations, so a list of `n` Phraser objects looks for `n+1` word-length MWEs.
        The first object in the list might transform neighbouring occurrences of "Harry" and "James" to "Harry_James",
        and the second object in the list transform "Harry_James" and "Potter" to "Harry_James_Potter"
        """

        self.n_gram_phrasers = n_gram_phrasers
        self.sentences = MySentences(files=files)

    def __iter__(self):

        for s in self.sentences:
            for phraser in self.n_gram_phrasers:
                # Convert adjacent words into collocations, incrementally if more than one Phraser
                s = phraser[s]
            yield s


class Vocabulary(object):
    """
    Class to generate and store a vocabulary over a training document corpus.
    After initialized, basically just a wrapper for a vocab dictionary.
    Stores a unique integer with each vocabulary word, making it easy to generate one-hot vectors,
    or use as row/column indices in a word-word matrix or term-document matrices.
    """

    def __init__(self, files, min_count=1):
        """
        `files` is a list of  paths to text files that constitute our training document collection.

        `min_count` the number of times a word must occur across corpus for it to be considered an entry in the vocab.
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

        print("Building vocab ...")

        sentences = MySentences(files=files)

        # Initialize a dictionary, mapping from word to its count in training corpus.
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

        print("Constructed vocabulary.\n")

        return new_vocab


class MyTargetContextPairs(object):
    """
    Iterates over text files to produce pairs of (target word, context words) for training (neural-network) LMs.
    Use MySentences class to iterate over tokens and sentences,
    and Vocabulary class to remove rare words (before creating context windows).
    """

    def __init__(self, files, vocab, before_window=2, after_window=2):
        """
        `files` is a list of paths to text files to return (target, context) word pairs over.

        `min_count` is the minimum frequency of a token across the corpus to consider it in the vocabulary.
        This is used to initialize a vocabulary object over `files`: we ignore all tokens not in the vocabulary.
        We remove these tokens before creating context windows, narrowing the distance between legitimate tokens.

        `before_window` and `after_window` are the number of words either side of the target word to look for.
        Usually the window is symmetric either side, if training word2vec models for example,
        but if we're training for language modelling, we only use words before to predict the target (next) word.
        """

        self.vocab = vocab
        self.after_window = after_window
        self.before_window = before_window
        self.sentences = MySentences(files=files)

    def __iter__(self):

        for sent in self.sentences:

            for i, target in enumerate(sent):
                # We yield one (target, context) pair for each target word
                # The context element in the pair is a list of words either side of the target word

                if target not in self.vocab:
                    continue

                # Keep looking for context words before target word until we either reach beginning of the sentence,
                # or we have more than we are looking for (specified by `self.before_window`)
                # If token not in vocabulary, we skip, ultimately increasing effective size of window.
                j = 1
                words_before = []
                while len(words_before) != self.before_window:

                    pos = i - j
                    if pos < 0:
                        # No tokens before beginning of sentence
                        break

                    context = sent[pos]
                    if context in self.vocab:
                        words_before.append(context)

                    j += 1

                # Do same for context words after target word
                j = 1
                words_after = []
                while len(words_after) != self.after_window:

                    pos = i + j
                    if pos >= len(sent):
                        # No tokens after end of sentence
                        break

                    context = sent[pos]
                    if context in self.vocab:
                        words_after.append(context)

                    j += 1

                yield (target, list(reversed(words_before)) + words_after)


def split_data(files, path_to_save_dir, fraction_train=0.8, fraction_test=0.1):
    """
    `files` is a list of paths to text files, holding all the lines of text we want to train some sort of model over.
    We split all the text into three categories: training data, test data, validation data.
    We have three files at the end storing all the lines of texts for these three files,
    saved under the directory pointed to be `path_to_save_dir`.

    `fraction_train` and `fraction_test` are the percentage of the data to split into training and testing, respectively
    However much is left (i.e. 1.0 - `fraction_test` - `fraction_train`) is used for validation data.
    """

    # If destination directory to save files does not exist, make it
    if not os.path.exists(path_to_save_dir):
        os.mkdir(path_to_save_dir)

    # Calculate indices at which to split data, e.g. 0 - 80% for training, 80% - 90 % testing, and the rest validating
    assert fraction_train + fraction_test <= 1.0
    train_split = fraction_train
    test_split = fraction_train + fraction_test

    # Calculate the path to each of the test, train, and validate files
    train_data_file = os.path.join(path_to_save_dir, 'train_data.txt')
    test_data_file = os.path.join(path_to_save_dir, 'test_data.txt')
    validate_data_file = os.path.join(path_to_save_dir, 'validate_data.txt')

    # Open a file pointer for each of the three files
    train_fp = open(train_data_file, 'w')
    test_fp = open(test_data_file, 'w')
    validate_fp = open(validate_data_file, 'w')

    for f in files:

        with open(f, 'r') as open_f:

            for line in open_f:

                # Generate a random real number between 0.0 and 1.0
                rand = random.randint(0, 100) / 100.0

                # Find out which bin it lies between in train-test-validate portions, and put in that file

                if 0.0 <= rand < train_split:
                    train_fp.write(line)
                    train_fp.write("\n")

                elif train_split <= rand < test_split:
                    test_fp.write(line)
                    test_fp.write("\n")

                else:  # test_split <= rand <= validate_split
                    validate_fp.write(line)
                    validate_fp.write("\n")

    # Close all file pointers
    train_fp.close()
    test_fp.close()
    validate_fp.close()

    # Return paths to files if we want to use them down the pipeline
    return train_data_file, test_data_file, validate_data_file


def cosine_similarity(vector1, vector2):
    """
    Returns (one of the better) measures of similarity between two vectors: the cosine of the angle between them.
    Both `vector1` and `vector2` are just a list of numbers.
    """

    return (np.dot(vector1, vector2))/(norm(vector1) * norm(vector2))


def most_similar(model, word, top_n=5, similarity_measure=cosine_similarity):
    """
    Returns the `top_n` most similar words to `word` under the vector embedding model defined by the `model` object.
    `model` has an attribute, called `matrix`, that is the actual matrix of numbers.
    `similarity_measure` is the way we are defining similarity, typically cosine similarity.
    """

    # The `vocab` attribute of `model` is a map from word to unique integer, corresponding to the row index in matrix
    vocab = model.vocab
    matrix = model.matrix

    if word not in vocab:
        raise WordNotInVocabulary(word)

    # Get the corresponding row vector for word
    vector = matrix[vocab[word]]

    # `best_seen` is a list of tuples of form: (some word, similarity of it to `word`)
    # We keep this list sorted at all times, so to insert a new word is very simple:
    # Check if new word has a similarity greater than the word with the smallest index; if so, replace, and re-sort
    best_seen = []

    for other_word in vocab:

        if other_word == word:
            # A word and itself have a similarity of 1.0; not very useful
            continue

        other_vector = matrix[vocab[other_word]]
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


characters = [
    'harry',
    'ron',
    'neville',
    'hermione',
    'ginny',
    'luna'
]


teachers = [
    'dumbledore',
    'slughorn',
    'snape',
    'lupin',
    'quirrell',
    'umbridge',
    'trelawney',
    'mcgonagall',
    'lockhart'
]


animals = [
    'hedwig',
    'crookshanks',
    'fang',
    'scabbers'
]


houses = [
    'gryffindor',
    'ravenclaw',
    'slytherin',
    'hufflepuff'
]


spells = [
    'lumos',
    'reparo',
    'expelliarmus',
    'stupefy',
    'crucio',
    'accio',
    'protego',
    'impedimenta',
    'petrificus',
    'levicorpus',
    'imperio'
]
