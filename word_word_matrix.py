import os
import math
import pickle
import argparse
import numpy as np
from utils import MySentences, Vocabulary, WordNotInVocabulary


class WordWordMatrix(object):
    """
    Word-word, or term-context, matrix of |V|x|V| dimensionality.
    Each cell represents number of times row (target) word "co-occurs" in some context with column (context) word
    in the training corpus.
    By "co-occur" we might mean occur in the same document, but for us we mean within some customizable window size,
    (up to sentence boundaries).
    """

    def __init__(self, files, window_size=5, min_count=1, vocab=None, matrix=None):
        """
        `files` is a list of paths to text files that represent our document collection.

        `window_size` is how far to look either side of the row (target) word for co-occurring column (context) words.
         Larger window sizes favor more semantic/topical relationships;
         smaller window sizes favor more syntactic/functional relationships.

        `min_count` the number of times a word must occur across corpus for it to be considered an entry in the vocab.

        `vocab` and `matrix` can be non-None if we are loading from a file,
        otherwise we initialize them from `files` and `min_count`
        """

        self.files = sorted(files)
        self.min_count = min_count
        self.window_size = window_size

        if matrix is None and vocab is None:
            self.vocab = Vocabulary(files=self.files, min_count=self.min_count)
            self.matrix = self._init_matrix()
        else:
            self.vocab = vocab
            self.matrix = matrix

    @staticmethod
    def load(path_to_model):
        """
        Load a pickle saved model from file path `path_to_model`
        """

        with open(path_to_model, 'r') as open_f:
            model = pickle.load(open_f)

        return WordWordMatrix(
            files=model['files'],
            window_size=model['window_size'],
            min_count=model['min_count'],
            vocab=model['vocab'],
            matrix=model['matrix']
        )

    def save(self, path_to_model):
        """
        Save as a pickle model under file path `path_to_model`
        """

        model = {
            'files': self.files,
            'window_size': self.window_size,
            'min_count': self.min_count,
            'vocab': self.vocab,
            'matrix': self.matrix
        }

        with open(path_to_model, 'w') as open_f:
            pickle.dump(model, open_f)

    def __getitem__(self, item):

        return self.get_word_vector(word=item)

    def get_word_vector(self, word):
        """
        Get the corresponding row vector for this `word`.
        The unique integer it is associated with in `self.vocab` is the index of the row to look up.
        """

        if word not in self.vocab:
            raise WordNotInVocabulary(word)

        return self.matrix[self.vocab[word]]

    def _init_matrix(self):
        """
        Returns a |V|x|V| matrix m, where m[i][j] is the number of times the word corresponding to index i in the vocab
        co-occurs with the word corresponding to index j in the vocab, in a window specified by `self.window_size`
        """

        # matrix[i] returns the ith row; matrix [i][j] returns the value of the cell at the ith row and jth column
        # Initialize all counts to 0.0 floats
        matrix = [[0.0 for col in range(len(self.vocab))] for row in range(len(self.vocab))]

        sentences = MySentences(files=self.files)

        for sent in sentences:

            for i, target in enumerate(sent):

                if target not in self.vocab:
                    # No row in the matrix exists for this token
                    continue

                row = self.vocab[target]  # Row index in the matrix corresponding to `target`

                for j in range(1, self.window_size+1):
                    # For each context word before target word within window size ...

                    pos = i - j
                    if pos < 0:
                        # No tokens before beginning of sentence
                        break

                    context = sent[pos]
                    if context not in self.vocab:
                        # No column in the matrix exists for this token
                        continue

                    col = self.vocab[context]  # Column index corresponding to `context`
                    matrix[row][col] += 1.0

                for j in range(1, self.window_size+1):
                    # For each context word after target word within window size ...

                    pos = i + j
                    if pos >= len(sent):
                        # No tokens after end of sentence
                        break

                    context = sent[pos]
                    if context not in self.vocab:
                        # No column in the matrix exists for this token
                        continue

                    col = self.vocab[context]  # Column index corresponding to `context`
                    matrix[row][col] += 1.0

        return matrix

    def ppmi_reweight_matrix(self, smooth_distribution=1.0):
        """
        Re-weights cell values based on PPMI weighting scheme.
        This is important because raw frequencies are very skewed and not discriminative.
        For example, words like 'the', 'it', 'they' occur frequently in the context of all sorts of words,
        but don't really have any discriminating power to inform us what the word actually means.

        Because PMI favors very rare words, we smooth unigram context distribution.
        `smooth_distribution` controls this amount of smoothing. Default (no smoothing) = 1.0. Most common value = 0.75
        """

        rows = len(self.matrix)
        cols = len(self.matrix[0])  # Assumes rectangular matrix

        # Initialize a zero-matrix the same size as the original `self.matrix`
        weighted_matrix = [[0.0 for col in range(cols)] for row in range(rows)]

        # Normalization sum over all cells
        N = sum([col for row in self.matrix for col in row])

        # Probability of target word, keyed row-index: probability = sum of row's elements / normalizing N
        word_probability = [sum(self.matrix[i])/N for i in range(rows)]

        # Sum over context counts (i.e. sum of columns), to the power of `smooth_distribution`
        smoothed_context_counts = 0.0
        for j in range(cols):
            smoothed_context_counts += sum([row[j] for row in self.matrix]) ** smooth_distribution

        # Probability of target word, keyed by column-index: probability = sum of column's elements / normalizing N
        # (potentially smoothed)
        context_probability = [
            (sum([row[j] for row in self.matrix]) ** smooth_distribution)/smoothed_context_counts
            for j in range(cols)
        ]

        for i in range(rows):
            for j in range(cols):

                p_ij = self.matrix[i][j]/N

                if p_ij == 0.0:
                    # Ignore, and keep initialized 0.0 value in entry
                    continue

                p_i = word_probability[i]
                p_j = context_probability[j]

                pmi = math.log(p_ij/(p_i * p_j), 2)

                weighted_matrix[i][j] = max(pmi, 0.0)  # Positive PMI (PPMI)

        self.matrix = weighted_matrix

    def svd_truncate(self, k_dim=250):
        """
        A form of dimensionality-reduction from our sparse, long, co-occurrence based count vectors.
        Decompose (possibly weighted) co-occurrence matrix `self.matrix` into product of three matrices:
          - W, an orthonormal matrix of left singular vectors as columns.
          - Sigma, a diagonal matrix of singular values.
          - C (transpose), an orthonormal matrix of right singular vectors as columns.

        We then truncate the rows of W to `k_dim`, and use the resulting vectors as our short, dense word vectors.
        """

        # We want reduced SVD, not full SVD.
        # Reduced SVD ensures width of W equals height of Sigma,
        # by discarding orthonormal column vectors from W that do not correspond to diagonal values
        w, sigma, c_transpose = np.linalg.svd(np.array(self.matrix), full_matrices=False)

        # Truncate each row in W, and return resulting matrix (list of list of numbers)
        self.matrix = [row[:k_dim] for row in w.tolist()]


parser = argparse.ArgumentParser(description='Script for training a term-context matrix over Harry Potter.')
parser.add_argument('-w', '--window', default=5, type=int, help='Size of window to base co-occurrence counts on.')
parser.add_argument('-c', '--count', default=20, type=int, help='Minimum count of frequency for valid tokens.')
parser.add_argument('-s', '--save', required=True, type=str, help='Path to save resulting model.')


if __name__ == '__main__':

    args = parser.parse_args()

    potter_files = [os.path.join('data/', x) for x in os.listdir('data/')]

    word_word_matrix = WordWordMatrix(
        files=potter_files,
        window_size=args.window,
        min_count=args.count
    )

    word_word_matrix.save(
        path_to_model=args.save
    )
