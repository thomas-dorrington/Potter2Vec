import os
import pickle
from utils import MySentences, Vocabulary, WordNotInVocabulary, DocumentNotInCollection


class TermDocumentMatrix(object):
    """
    Term-Document matrix of |V|x|D| dimensionality,
    for |V| the vocab size, and |D| the number of documents in collection.
    Each row corresponds to a word in the vocab, and each column a document in the document training collection.
    Thus, each cell represents the number of times the word (specified by row) occurs in the document
    (specified by the column).
    This allows us to have vectors for both documents (the columns: 1x|V|) and words (the rows: 1x|D|).
    Originated in the task of document retrieval, from the field of Information Retrieval.
    """

    def __init__(self, files, min_count=1, vocab=None, matrix=None):
        """
        `files` is a list of paths to text files that represent our document collection.

        `min_count` the number of times a word must occur in all the text for it to be considered an entry in the vocab.

        `vocab` and `matrix` can be non-None if we are loading from a file,
        otherwise we initialize them from `files` and `min_count`
        """

        self.files = files
        self.min_count = min_count

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

        return TermDocumentMatrix(
            files=model['files'],
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
            'min_count': self.min_count,
            'vocab': self.vocab,
            'matrix': self.matrix
        }

        with open(path_to_model, 'w') as open_f:
            pickle.dump(model, open_f)

    def __getitem__(self, item):
        """
        Term-Document matrices are more often than not used to get vectors for documents.
        We are assuming `item` is a document we wish to retrieve its corresponding column vector for,
        but just in case we try treating it as a word, catching its thrown exception to discard if it failed.
        """

        try:
            self.get_word_vector(word=item)
        except WordNotInVocabulary:
            pass

        self.get_document_vector(f=item)

    def get_word_vector(self, word):
        """
        Get the corresponding row vector for this `word`
        The unique integer it is associated with in `self.vocab` is the index of the row to look up.
        """

        if word not in self.vocab:
            raise WordNotInVocabulary(word)

        return self.matrix[self.vocab[word]]

    def get_document_vector(self, f):
        """
        Get the corresponding column vector for this `file`.
        It must match excctly the file in `self.files`, so if it was originally trained with relative paths
        to where the current directory was for the script, it must match that.
        The index of `file` in `self.files` is the index of the column to look up.
        """

        if f not in self.files:
            raise DocumentNotInCollection(f)

        col = self.files.index(f)

        return [self.matrix[row][col] for row in range(len(self.vocab))]

    def _init_matrix(self):
        """
        Returns a |V|x|D| matrix m, where m[i][j] is the number of times the word corresponding to index i in the vocab
        occurs in the document corresponding to index j.
        """

        # matrix[i] returns the ith row; matrix [i][j] returns the value of the cell at the ith row and jth column
        # Initialize all counts to 0.0 floats
        # The value at each cell is the number of times the word (row i) occurs in the document (column j)
        matrix = [[0.0 for col in range(len(self.files))] for row in range(len(self.vocab))]

        for col, f in enumerate(self.files):

            # Initialize a sentence iterator, associated just with this file
            sentences = MySentences(files=[f])

            for sent in sentences:

                for tok in sent:

                    if tok not in self.vocab:
                        # No row in the matrix exists for this token
                        continue

                    row = self.vocab[tok]  # Corresponding row in matrix

                    matrix[row][col] += 1.0

        return matrix


if __name__ == '__main__':

    potter_files = [os.path.join('data/', x) for x in os.listdir('data/')]

    term_doc_matrix = TermDocumentMatrix(files=potter_files, min_count=20)
    term_doc_matrix.save('test.bin')
