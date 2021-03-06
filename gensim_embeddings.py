import os
import json
import pickle
import argparse
from matplotlib import pyplot
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from utils import MySentences, MyNGrams
from gensim.models.phrases import Phraser, Phrases
from scipy.cluster.hierarchy import dendrogram, linkage


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
                 phrasers=None,
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

        It is possible to load pre-trained models from disk, in which case `model` and `phrasers` will be non-None.
        """

        self.files = files
        self.n_iter = n_iter
        self.window = window
        self.workers = workers
        self.algorithm = algorithm
        self.min_count = min_count
        self.vector_size = vector_size
        self.phrases_size = phrases_size
        self.negative_samples = negative_samples

        if phrasers is None and model is None:
            self.model, self.phrasers = self._train_model()
        else:
            self.model = model
            self.phrasers = phrasers

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
        # If we want to learn vectors for multi-word phrases, we will assign a different iterator to this variable name.
        sents = MySentences(files=self.files)

        # List to accumulate Phraser objects that successively look for longer collocations.
        # They need to be applied one after the other, starting at the beginning, to build longer chains of words.
        phrasers = []

        for i in range(self.phrases_size):

            print("\nLooking for length %s collocations ..." % str(i+2))

            phrases = Phrases(sentences=sents, min_count=5, threshold=10.0)
            phrasers.append(Phraser(phrases_model=phrases))

            sents = MyNGrams(files=self.files, n_gram_phrasers=phrasers)

        print("\nTraining model for %s iterations ..." % str(self.n_iter))

        model = Word2Vec(
            sentences=sents,                 # Sentence iteration
            iter=self.n_iter,                # How many epochs to train for
            window=self.window,              # Size of context window to take tokens from for training
            min_count=self.min_count,        # Only consider tokens with frequency greater than this
            workers=self.workers,            # How many threads to train model
            negative=self.negative_samples,  # How many negative sample to use
            size=self.vector_size,           # Size of resulting embeddings
            sg=1 if self.algorithm == 'skip-gram' else 0  # Skip-gram or CBOW
        )

        return model, phrasers

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
            'phrasers': self.phrasers,
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
            phrasers=model['phrasers'],
            model=model['model']
        )

    def pca_plot(self, vocab=None):
        """
        Plots a custom vocabulary list using PCA to visualize in 2D.
        If no vocabulary past in, default to whole vocab of model; this takes a while!
        """

        vocab = vocab if vocab is not None else self.model.wv.vocab

        X = self.model[vocab]
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)

        # Create a scatter plot of the projection
        pyplot.scatter(result[:, 0], result[:, 1])
        for i, word in enumerate(vocab):
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

        # Display scatter plot
        pyplot.show()

    def hierarchical_cluster_plot(self, words):
        """
        Plots a hierarchical cluster plot for a subset of `words`
        """

        matrix = [self.model[wd] for wd in words]

        index2wd = {i: wd for i, wd in enumerate(words)}

        l = linkage(matrix, method='complete', metric='seuclidean')

        # Calculate full dendrogram
        pyplot.figure(figsize=(25, 10))
        pyplot.ylabel('word')
        pyplot.xlabel('distance')

        dendrogram(
            l,
            orientation='left',
            leaf_font_size=10,
            leaf_label_func=lambda v: str(index2wd[v])
        )

        # Display plot
        pyplot.show()


parser = argparse.ArgumentParser(description='Script for training Word2Vec (skip-gram) model over Harry Potter books.')
parser.add_argument('--algorithm', type=str, default='skip-gram', help='"skip-gram" or "cbow"')
parser.add_argument('--window', default=5, type=int, help='Size of context window to take training tokens from.')
parser.add_argument('--iter', default=20, type=int, help='How many iterations to train model over.')
parser.add_argument('--min_count', default=5, type=int, help='Minimum count of frequency for valid tokens.')
parser.add_argument('--dim', default=100, type=int, help='Size of resulting vector embeddings.')
parser.add_argument('--phrases', type=int, default=0, help='How large are multi-word token phrases.')
parser.add_argument('--negative', type=int, default=5, help='How many negative examples per target word.')
parser.add_argument('--threads', type=int, default=4, help='How many worker threads to use during training.')
parser.add_argument('--save', type=str, required=True, help='Path to save resulting models.')


if __name__ == '__main__':

    args = parser.parse_args()

    potter_files = [os.path.join('data/', x) for x in os.listdir('data/')]

    embeddings = GensimEmbeddings(
        files=potter_files,
        algorithm=args.algorithm,
        window=args.window,
        n_iter=args.iter,
        min_count=args.min_count,
        vector_size=args.dim,
        phrases_size=args.phrases,
        negative_samples=args.negative,
        workers=args.threads
    )

    embeddings.save(
        path_to_save=args.save
    )

    test_vocab = [
        "gryffindor",
        "slytherin",
        "hufflepuff",
        "ravenclaw",
        "hermione",
        "harry",
        "avada_kedavra",
        "ron",
        "hagrid",
        "malfoy",
        "dumbledore",
        "snape",
        "broomstick",
        "magic",
        "witch",
        "wizard",
        "muggle",
        "witches",
        "diagon_alley",
        "knockturn_alley",
        "hedwig",
        "scabbers",
        "buckbeak",
        "charm",
        "spell",
        "curse",
        "lumos",
        "accio",
        "death_eater"
    ]
