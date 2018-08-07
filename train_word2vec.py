import os
import argparse
from gensim.models import Word2Vec
from utils import MySentences, MyNGrams
from gensim.models.phrases import Phraser, Phrases


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
