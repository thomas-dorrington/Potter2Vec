import os
import math
import pickle
import argparse
import numpy as np
from numpy.linalg import svd
from utils import MySentences, Vocabulary


class WordWordMatrix(object):
    """
    Word-Word, or Term-Term, matrix of |V|x|V| dimensionality.
    Each cell represents number of times row (target) word "co-occurs" in some context with column (context) word
    in the training corpus.
    By "co-occur" we might mean occur in the same document, but for us we mean within some customizable window size,
    (up to sentence boundaries).
    """

    def __init__(self, files, window_size=5, min_count=1):
        """
        `sentences` is an iterable object, that yields sentences over our training corpus.
         We iterate over the sentences twice: once to build vocab, once to build co-occurrence matrix.

        `window_size` is how far to look either side of the row (target) word for co-occuring column (context) words.
         Large window sizes favor semantic relationships; smaller window sizes syntactic relationships.

        `min_count` is the number of times a word must occur for it to be considered an entry in the vocab.
         Larger number significantly speeds up run-time.
        """

        self.window_size = window_size
        self.vocab = Vocabulary(files=files, min_count=min_count)  # Initialize a map from word type to integer
        self.matrix = self._init_matrix(files=files)               # Initialize a co-occurrence word-word matrix

    def _init_matrix(self, files):
        """
        Returns a |V|x|V| matrix m, where m[i][j] is the number of times the word corresponding to index i in the vocab
        co-occurs with the word corresponding to index j in the vocab.
        """

        sentences = MySentences(files=files)

        # matrix[i] returns the ith row; matrix [i][j] returns the value of the cell at the ith row and jth column
        # Initialize all counts to 0.0 floats
        # The value at each cell is the number of times the word (row i) co-occurs with the other word (column j)
        matrix = [[0.0 for col in range(len(self.vocab))] for row in range(len(self.vocab))]

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


class TFIDFMatrix(object):

    def __init__(self, matrix):

        self.matrix = self._reweight_matrix(matrix=matrix.matrix)

    def _reweight_matrix(self, matrix):

        return matrix


class PPMIMatrix(object):
    """
    Takes a Matrix object, e.g. word-word matrix or term-document matrix,
    and re-weights cell values based on PPMI weighting scheme.
    This is important because raw frequencies are very skewed and not discriminative.
    For example, words like 'the', 'it', 'they' occur frequently in the context of all sorts of words,
    but don't really have any discriminating power to inform us what the word actually means.
    """

    def __init__(self, matrix):
        """
        `matrix` is a Matrix object, with a field also called `matrix` that is the actual array of array of numbers.
        """

        self.matrix = self._reweight_matrix(matrix=matrix.matrix)

    def _reweight_matrix(self, matrix):
        """
        Takes an array of array of numbers to re-weight based on PPMI weighting scheme
        """

        rows = len(matrix)
        cols = len(matrix[0])  # Assumes rectangular matrix

        # Initialize a zero-matrix the same size as the original `matrix`
        weighted_matrix = [[0.0 for col in range(cols)] for row in range(rows)]

        # Normalization sum over all cells
        normalization_constant = sum([col for row in matrix for col in row])

        # Pre-compute the normalization constant per-row: map from row number to sum of that rows' elements
        row_normalization = {i: sum(matrix[i])/normalization_constant for i in range(rows)}

        # Pre-compute normalization constant per-column: map from column number to sum of that columns' elements
        col_normalization = {j: sum([row[j] for row in matrix])/normalization_constant for j in range(cols)}

        for i in range(rows):
            for j in range(cols):

                p_i_j = matrix[i][j]/normalization_constant

                if p_i_j == 0.0:
                    # Ignore, and keep initialized 0.0 value in entry
                    continue

                p_i = row_normalization[i]
                p_j = col_normalization[j]

                pmi = math.log(p_i_j/(p_i * p_j), 2)

                weighted_matrix[i][j] = pmi if pmi > 0.0 else 0.0  # Positive PMI (PPMI)

        return weighted_matrix


class SVDMatrix(object):

    def __init__(self, matrix, k_dim=250):

        self.k_dim = k_dim
        self.matrix = self._svd_matrix(matrix=matrix.matrix)

    def _svd_matrix(self, matrix):

        w, sigma, c_transpose = svd(np.array(matrix))
        return [row[:self.k_dim] for row in w.tolist()]


def compare_embedding_matrices(matrix, ppmi_matrix, svd_matrix):
    """
    Takes three word-embedding matrices:
      - `matrix` is our raw-frequency co-occurrence count matrix
      - `ppmi_matrix` is our PPMI weighted co-occurrence count matrix
      - `svd_matrix` is the top-k dimensions from our SVD of `ppmi_matrix`

    We look at what the most similar words for a given set list are, to see which gives the best reasonable performance
    Warning: needs a bit of Harry Potter knowledge here
    """

    words_to_compare = [
        'harry',
        'wand',
        'spell',
        'school',
        'voldemort',
        'wizard'
    ]

    for wd in words_to_compare:

        print("### Most similar words to '%s' with raw frequencies" % wd)
        for similar in most_similar(matrix=matrix, vocab=matrix.vocab, word=wd):
            print(similar)
        print

        print("### Most similar words to '%s' with PPMI weighting" % wd)
        for similar in most_similar(matrix=ppmi_matrix, vocab=matrix.vocab, word=wd):
            print(similar)
        print

        print("### Most similar wards to '%s' with SVD PPMI weighting" % wd)
        for similar in most_similar(matrix=svd_matrix, vocab=matrix.vocab, word=wd):
            print(similar)
        print


parser = argparse.ArgumentParser(description='Script for training a PPMI-weighted Word-Word matrix over Harry Potter.')
parser.add_argument('-w', '--window_size', type=int, default=5,
                    help='Size of window to based co-occurrence counts on.')
parser.add_argument('-c', '--min_count', type=int, default=50,
                    help='Minimum number of occurrence of tokens before counting as part of vocabulary.')
parser.add_argument('-d', '--dir', type=str, default="",
                    help='Directory to save resulting matrices. If name clashes, overwrites. If missing, does not save')
parser.add_argument('-k', '--k_dim', type=int, default=250,
                    help='Top k-dimensions from SVD to keep; this is the size of our resulting embeddings.')

if __name__ == '__main__':

    args = parser.parse_args()

    potter_files = [os.path.join('data/', x) for x in os.listdir('data/')]

    matrix = WordWordMatrix(files=potter_files, window_size=args.window_size, min_count=args.min_count)

    ppmi_matrix = PPMIMatrix(matrix=matrix)

    svd_matrix = SVDMatrix(matrix=matrix, k_dim=args.k_dim)

    compare_embedding_matrices(
        matrix=matrix,
        ppmi_matrix=ppmi_matrix,
        svd_matrix=svd_matrix
    )

    if args.dir:
        # If path to directory to save the non-default (i.e. empty string), save.

        if not os.path.exists(args.dir):
            os.mkdir(args.dir)

        with open(os.path.join(args.dir, 'matrix.bin'), 'w') as open_f:
            pickle.dump(matrix, open_f)

        with open(os.path.join(args.dir, 'ppmi_matrix.bin'), 'w') as open_f:
            pickle.dump(ppmi_matrix, open_f)

        with open(os.path.join(args.dir, 'svd_matrix.bin'), 'w') as open_f:
            pickle.dump(svd_matrix, open_f)
