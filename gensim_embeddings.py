import os
import json
import pickle
import argparse
from gensim.models import Word2Vec
from utils import MySentences, MyNGrams
from gensim.models.phrases import Phraser, Phrases


class GensimEmbeddings(object):
    """
    Wrapper class for generating word embeddings over a series of text files, using Gensim Python library.
    Makes the API a bit clearer & customized to our needs, and add additional methods.
    """

    def __init__(self,
                 files,
                 algorithm='skip-gram',
                 window=5,
                 n_iter=20,
                 min_count=5,
                 vector_size=100,
                 phrases_size=0,
                 negative_samples=5,
                 workers=4,
                 verbose=True,
                 phraser=None,
                 model=None):
        """
        `files` is a list of (relative) paths to a series of text files to generate embeddings over.

        `algorithm` is either "skip-gram" or "cbow".

        `window` is the size of the context window to use in generating training data.
        For skip-gram this means, for each target (centre) word, we generate 2 * `window` other words
        in a symmetric window either side, and try and predict each one from the target word in separate tasks.
        Larger window sizes lead to more topical, or semantic, similarities among words;
        Smaller window sizes lead to more functional, or syntactic, similarities among words.
        Technically, this is the maximum window size; we use a dynamic window size, sampled uniformly from 1 to `window`
        This is a faster way to simulate the technique of weighting words that are closer to the target word
        more than those further away (faster because we on average we have less training examples to train for).

        `workers` is how many threads to spawn in training, to speed up neural network training.
        Note, this will NOT speed up how fast the vocabulary is generated,
        or how fast multi-word tokens are determined, only embedding training;
        Both these tasks use only 1 thread.
        More workers will not improve speed (in fact, probably make it slower) if data pre-processing
        and sentence iteration is the bottleneck.

        `n_iter` is the number of iterations (i.e. epochs) to train model over.

        We ignore all words with a frequency less than `min_count`.

        `vector_size` is the size of the resulting vector embeddings.

        To handle multi-word expressions, e.g. "harry_potter" or "expecto_patronum",
        we can consider how often the tokens occur together based on the frequency of the individual tokens.
        `phrases_size` tells us how many tokens to consider as making up multi-word phrases.
        0 means we use no phrasing, just the original (unigram) tokens.
        1 means we consider phrases as consisting of two tokens, and so on.
        In summary, `phrases_size`+1 is the max length any resulting token-sequence can be.
        This means we can give an actual vector embedding to the token "harry_potter",
        rather than try and infer the meaning from the two individual embeddings "harry" and "potter".

        We default to skip-gram with negative sampling (NEG) model, which while it might take longer to train than CBOW,
        generally produces more accurate results.
        During skip-gram training, for each target word and context word pair, we have one positive example:
        predict the context word from the target word.
        We also generate another `negative_samples` to use as negative examples in training.
        This is a more efficient way of training than a full-blown soft-max layer.
        Hierarchical soft-max would be the alternative, but we follow blog's material and use NEG.

        It is possible to load one of these pre-trained models from disk,
        in which case both `model` and `phraser` will be None.
        If `model` is not None, but `phraser` is, that means we trained a model just over single tokens, not phrases.
        """

        self.files = files
        self.n_iter = n_iter
        self.window = window
        self.workers = workers
        self.verbose = verbose
        self.algorithm = algorithm
        self.min_count = min_count
        self.vector_size = vector_size
        self.phrases_size = phrases_size
        self.negative_samples = negative_samples

        if phraser is None and model is None:
            self.model, self.phraser = self._train_model()
        else:
            self.model = model
            self.phraser = phraser

    def __repr__(self):
        """
        String representation of class for pretty printing
        """

        return json.dumps(
            {
                'Algorithm': self.algorithm,
                'Vector Size': self.vector_size,
                'Window Size': self.window,
                'Minimum Count': self.min_count,
                'Negative Samples': self.negative_samples,
                'No. of Iterations': self.n_iter,
                'Phrases Max Length': self.phrases_size
            },
            indent=4
        )

    def _train_model(self):

        # `sents` is the iterator we will eventually pass to the word2vec algorithm to generate word embeddings over
        # If we want to learn vectors for multi-word phrases, we will assign a different iterator to this variable name
        sents = MySentences(txt_files=self.files, verbose=self.verbose)

        phraser = None
        for i in range(self.phrases_size):

            if self.verbose:
                print("Looking for %s length token phrases ..." % str(i+2))

            phrases = Phrases(sentences=sents, min_count=5, threshold=10.0)
            phraser = Phraser(phrases_model=phrases)  # Much smaller and more efficient version of Phrases class
            sents = MyNGrams(txt_files=self.files, n_gram_phraser=phraser, verbose=self.verbose)

        model = Word2Vec(
            sentences=sents,                 # Sentence iteration
            iter=self.n_iter,                # How many epochs to train for
            window=self.window,              # Size of context window to take tokens from for training
            min_count=self.min_count,        # Only consider tokens with frequency greater than this
            workers=self.workers,            # How many threads to train model
            negative=self.negative_samples,  # How many negative sample to use
            size=self.vector_size,           # Size of resulting embeddings
            sg=1 if self.algorithm == 'skip-gram' else 0
        )

        return model, phraser

    def save(self, path_to_save):

        model = {
            'files': self.files,
            'algorithm': self.algorithm,
            'window': self.window,
            'n_iter': self.n_iter,
            'min_count': self.min_count,
            'vector_size': self.vector_size,
            'phrases_size': self.phrases_size,
            'negative_samples': self.negative_samples,
            'workers': self.workers,
            'verbose': self.verbose,
            'phraser': self.phraser,
            'model': self.model
        }

        with open(path_to_save, 'w') as open_f:
            pickle.dump(model, open_f)

    @staticmethod
    def load(path_to_model):

        with open(path_to_model, 'r') as open_f:
            model = pickle.load(open_f)

        return GensimEmbeddings(
            files=model['files'],
            algorithm=model['algorithm'],
            window=model['window'],
            n_iter=model['n_iter'],
            min_count=model['min_count'],
            vector_size=model['vector_size'],
            phrases_size=model['phrases_size'],
            negative_samples=model['negative_samples'],
            workers=model['workers'],
            verbose=model['verbose'],
            phraser=model['phraser'],
            model=model['model']
        )


parser = argparse.ArgumentParser(description='Script for training Word2Vec (skip-gram) model over Harry Potter books.')
parser.add_argument('-a', '--algorithm', type=str, default='skip-gram', help='"skip-gram" or "cbow"')
parser.add_argument('-w', '--window', default=5, type=int, help='Size of context window to take training tokens from.')
parser.add_argument('-i', '--iter', default=20, type=int, help='How many iterations to train model over.')
parser.add_argument('-c', '--count', default=5, type=int, help='Minimum count of frequency for valid tokens.')
parser.add_argument('-d', '--dim', default=100, type=int, help='Size of resulting vector embeddings.')
parser.add_argument('-p', '--phrases', type=int, default=0, help='How large are multi-word token phrases.')
parser.add_argument('-n', '--negative', type=int, default=5, help='How many negative examples per target word.')
parser.add_argument('-t', '--threads', type=int, default=4, help='How many worker threads to use during training.')
parser.add_argument('-v', '--verbose', type=bool, default=True, help='Print logging information during training.')
parser.add_argument('-s', '--save', type=str, required=True, help='Path to save resulting models.')


if __name__ == '__main__':

    args = parser.parse_args()

    potter_files = [os.path.join('data/', x) for x in os.listdir('data/')]

    embeddings = GensimEmbeddings(
        files=potter_files,
        algorithm=args.algorithm,
        window=args.window,
        n_iter=args.iter,
        min_count=args.count,
        vector_size=args.dim,
        phrases_size=args.phrases,
        negative_samples=args.negative,
        workers=args.threads,
        verbose=args.verbose
    )

    embeddings.save(
        path_to_save=args.save
    )

