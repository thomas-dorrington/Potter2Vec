import json
import pickle
import theano
import argparse
import numpy as np
import theano.tensor as T
from theano.printing import pydotprint
from theano.tensor.nnet import softmax
from utils import Vocabulary, MyTargetContextPairs, split_data


def ReLU(z):

    return T.maximum(0.0, z)


def activation_to_string(fn):

    return {
        ReLU: 'ReLU'
    }[fn]


def layer_class_from_string(s):

    return {
        'EmbeddingLayer': EmbeddingLayer,
        'FullyConnectedLayer': FullyConnectedLayer,
        'SoftmaxLayer': SoftmaxLayer
    }[s]


def activation_from_string(s):

    return {
        'ReLU': ReLU
    }[s]


class Network(object):
    """
     Overall network object initializes and stores computation graph for neural network.
     Allows us to train such a network in a supervised fashion, with methods for loading and saving models.
     """

    def __init__(self, layers, vocab, mini_batch_size):
        """
        Takes a list of `layers` describing the network architecture (e.g. EmbeddingLayer, SoftmaxLayer, etc.)

        `mini_batch_size` is the number of supervised examples to use in each mini batch during training.
        Once done training, we can only predict on `mini_batch_size` number of examples at a time.
        To predict on 1 examples at a time, say, would need to save model and load it again with a different batch size,
        so the tensor dimensions in the computation graph are appropriate.

        We save a `vocab` object along side the network, so we know how to embed words into their one-hot vectors,
        and can ultimately go all the way from text to predicting the next word.
        """

        self.vocab = vocab
        self.layers = layers
        self.mini_batch_size = mini_batch_size

        # Extract all the parameters in the network we want to learn
        self.params = [param for layer in self.layers for param in layer.params]

        # Symbolically set up networks input and output
        self.x = T.matrix("x")
        self.y = T.ivector("y")

        # Set the first layers input to our symbolic network input `self.x`
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.mini_batch_size)

        # Propagate each layers' output to the next layers input
        for j in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output, self.mini_batch_size)

        # Final output of network is the output of final layer
        self.output = self.layers[-1].output

    @staticmethod
    def load(path_to_model, mini_batch_size=None):
        """
        Load a pickle saved model, stored in JSON format.

        If `mini_batch_size` is None, load the model with whatever batch size the original model was saved with.
        Otherwise, load the new model with `mini_batch_size`.
        This allows us to predict on 1 example at a time say, rather than whatever the original model was trained on.
        """

        with open(path_to_model, 'r') as open_f:
            model = pickle.load(open_f)

        return Network(
            layers=[layer_class_from_string(layer['type']).from_json(layer) for layer in model['layers']],
            vocab=Vocabulary.from_json(model['vocab']),
            mini_batch_size=mini_batch_size if mini_batch_size is not None else model['mini_batch_size']
        )

    def save(self, path_to_model):
        """
        Compiles model into JSON format, than saves as a pickle dump.
        """

        model = {
            'layers': [layer.to_json() for layer in self.layers],
            'vocab': self.vocab.to_json(),
            'mini_batch_size': self.mini_batch_size
        }

        with open(path_to_model, 'w') as open_f:
            pickle.dump(model, open_f)

    def __repr__(self):
        """"
        Pretty string representation of Network object
        """

        return json.dumps(
            [json.loads(layer.__repr__()) for layer in self.layers],
            indent=4
        )

    def plot(self, path_to_save):
        """
        Save an image of the network's computation graph to `path_to_save`
        """

        pydotprint(theano.function([self.x], self.output), path_to_save)

    @staticmethod
    def adam(cost, params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=1 - 1e-8):
        """
        Adam update rules for learning; improves over traditional SGD.
        Credit: https://gist.github.com/skaae/ae7225263ca8806868cb
        Based on: http://arxiv.org/pdf/1412.6980v4.pdf
        """

        grads = theano.grad(cost, params)  # Get gradients wrt. stochastic objective function
        t = theano.shared(np.float32(1.0))  # Initialize time-step to 1
        b1_t = b1 * gamma ** (t - 1.0)  # Decay the first moment running average coefficient

        updates = []
        for theta_previous, g in zip(params, grads):
            m_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX))
            v_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX))

            m = b1_t * m_previous + (1.0 - b1_t) * g
            v = b2 * v_previous + (1.0 - b2) * g ** 2.0
            m_hat = m / (1.0 - b1 ** t)
            v_hat = v / (1.0 - b2 ** t)

            theta = theta_previous - (learning_rate * m_hat) / (T.sqrt(v_hat) + e)

            updates.append((m_previous, m))
            updates.append((v_previous, v))
            updates.append((theta_previous, theta))

        updates.append((t, t + 1.0))

        return updates

    @staticmethod
    def SGD(cost, params, eta=0.001):
        """
        Traditional stochastic gradient descent.
        Simply move the parameters a small direction in the negative of gradient of cost function (wrt. that parameter)
        """

        grads = T.grad(cost, params)
        updates = [(param, param - eta * grad) for param, grad in zip(params, grads)]

        return updates

    def fit_iterator(self, train_iterator, test_iterator, validate_iterator, epochs, lmbda=0.0, update_method=adam):
        """
        Train `self` in iterator-like fashion.
        `train_iterator`, `test_iterator`, and `validate_iterator`, are all iterators that yield
        `self.mini_batch_size` number of (input, expected output) example pairs.
        Generating examples on the fly much more efficient than loading everything into memory at once.
        """

        # Define the (L2 regularized) cost function
        # We do not divide L2 regularization term by number of training batches; `lmbda` should be adjusted accordingly
        if lmbda > 0.0:
            l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])
            cost = self.layers[-1].cost(self) + 0.5 * lmbda * l2_norm_squared
        else:
            cost = self.layers[-1].cost(self)

        # Find symbolic update rules
        updates = update_method(cost, self.params)

        # Set up symbolic variables to hold mini-batch inputs & outputs
        x = T.matrix("x")
        y = T.ivector("y")

        # For a given mini-batch of inputs (`x`) and outputs (`y`), compute cost, and update parameters accordingly
        train_mb = theano.function(
            inputs=[x, y],
            outputs=cost,
            updates=updates,
            givens={self.x: x, self.y: y}
        )

        # Accuracy for a given mini-batch of inputs (`x`) and outputs (`y`)
        mb_accuracy = theano.function(
            inputs=[x, y],
            outputs=self.layers[-1].accuracy(self.y),
            givens={self.x: x, self.y: y}
        )

        # Store statistics during training
        best_epoch = 0.0
        best_test_accuracy = 0.0
        best_validation_accuracy = 0.0

        # Do the actual training
        for epoch in xrange(epochs):

            mini_batch_index = 0
            for train_x, train_y in train_iterator:
                # Each iteration of iterator yields one mini-batch

                if mini_batch_index % 1000 == 0:
                    print("Training mini-batch number {0} of epoch {1}".format(mini_batch_index, epoch))

                train_mb(train_x, train_y)
                mini_batch_index += 1

            print("Finished training epoch {0}, with {1} mini-batches\n".format(epoch, mini_batch_index))

            # After training from all mini-batches in one epoch, look at test & validation statistics
            validation_accuracy = np.mean([mb_accuracy(x, y) for x, y in validate_iterator])
            print("End of epoch {0}: validation accuracy of {1:.2%}".format(epoch, validation_accuracy))

            if validation_accuracy >= best_validation_accuracy:

                best_epoch = epoch
                best_validation_accuracy = validation_accuracy
                print("This is the best validation accuracy to date.")

                best_test_accuracy = np.mean([mb_accuracy(x, y) for x, y in test_iterator])
                print('The corresponding test accuracy for this epoch is {0:.2%}\n'.format(best_test_accuracy))

        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at end of epoch {1}".format(best_validation_accuracy,
                                                                                        best_epoch))
        print("Corresponding test accuracy of {0:.2%}".format(best_test_accuracy))


class EmbeddingLayer(object):

    def __init__(self, dim_size, vocab_size, context_size, w=None, learn_embeddings=True):
        """
        `dim_size` is the size we want the resulting word embeddings to have.

        `vocab_size` is used to determine the size of our one-hot vector inputs.

        `context_size` is how many previous words we are using to predict the next one.

        If `w` is non-None, we are loading a particular weight matrix, not randomly initializing one.
        This allows you to load saved networks, or even pre-trained word embedding models for fine-tuning.

        `learn_embeddings` is a Boolean flag telling us whether want to propagate errors back to this embedding matrix.
        Under normal circumstances, where we initialize a random embedding matrix, this will obviously be True,
        but we want not want to adjust our embedding matrix if we're loading a pre-trained word embedding model
        (say from word2vec or PPMI & SVD on a term-context matrix).
        Slight complication is that you cannot change this flag in-between saving & loading models;
        it will retain its value from whenever this class is first initialized; not too much of a problem.
        """

        self.dim_size = dim_size
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.learn_embeddings = learn_embeddings

        # The number of total input connections is the product of:
        # - How many elements are in the vocab (i.e. the size of the one-hot vectors)
        # - How many of these vectors there are (i.e. how many context words we're using to predict the next word)
        self.n_in = self.context_size * self.vocab_size

        # Initialize weight matrix (i.e. embedding matrix)
        # Each row of this matrix is a `dim_size`-dimensional vector,
        # corresponding to the word whose index in the vocabulary equals that row index
        if w is None:
            self.w = theano.shared(
                np.asarray(np.random.normal(loc=0.0, scale=np.sqrt(1.0/dim_size), size=(self.vocab_size, dim_size)),
                           dtype=theano.config.floatX),
                name='w',
                borrow=True
            )
        else:
            self.w = theano.shared(np.asarray(w, dtype=theano.config.floatX), name='w', borrow=True)

        # No bias or activation function.
        # Just acts as a look-up table to learn word embeddings by back-propagating errors all the way back
        # If we don't want to learn `self.w`, because we're using pre-trained embeddings we don't want to fine tune,
        # then this layer is set up with no learn-able parameters.
        self.params = [self.w] if self.learn_embeddings else []

    @staticmethod
    def from_json(layer):

        return EmbeddingLayer(
            dim_size=layer['dim_size'],
            vocab_size=layer['vocab_size'],
            context_size=layer['context_size'],
            w=layer['w'],
            learn_embeddings=layer['learn_embeddings']
        )

    def to_json(self):

        return {
            'type': 'EmbeddingLayer',
            'dim_size': self.dim_size,
            'vocab_size': self.vocab_size,
            'context_size': self.context_size,
            'w': self.w.get_value(),
            'learn_embeddings': self.learn_embeddings
        }

    def __repr__(self):

        return json.dumps(
            {
                'Type': 'Embedding Layer',
                'Learn Embeddings': self.learn_embeddings,
                'Embedding Matrix Shape': self.w.get_value().shape,
                'Vector Embedding Size': self.dim_size,
                'Vocabulary Size': self.vocab_size,
                'Context Size': self.context_size
            },
            indent=4
        )

    def set_inpt(self, inpt, mini_batch_size):

        self.inpt = inpt.reshape((mini_batch_size, self.n_in))

        # Split self.inpt into (mini_batch_size, vocab_size) sub-matrices, for however many context words there are;
        # Dot each with self.w (remember, we share same instantiation of embedding matrix across each one-hot vector);
        # Before concatenating each along axis=1
        self.output = T.concatenate(
            [T.dot(self.inpt[:, i * self.vocab_size:(i+1) * self.vocab_size], self.w)
             for i in range(self.context_size)],
            axis=1
        )


class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=ReLU, w=None, b=None):
        """
        Just a simple feed-forward, fully-connected, neural network layer,
        with `n_in` input connections, and `n_out` output connections.

        If `w` and `b` are non-None, we are loading a model from disk, otherwise randomly initialize.
        """

        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn

        # Initialize weights and biases

        if w is None:
            self.w = theano.shared(
                np.asarray(np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                           dtype=theano.config.floatX),
                name='w',
                borrow=True
            )
        else:
            self.w = theano.shared(np.asarray(w, dtype=theano.config.floatX), name='w', borrow=True)

        if b is None:
            self.b = theano.shared(
                np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)), dtype=theano.config.floatX),
                name='b',
                borrow=True
            )
        else:
            self.b = theano.shared(np.asarray(b, dtype=theano.config.floatX), name='b', borrow=True)

        self.params = [self.w, self.b]

    @staticmethod
    def from_json(layer):

        return FullyConnectedLayer(
            n_in=layer['n_in'],
            n_out=layer['n_out'],
            activation_fn=activation_from_string(layer['activation_fn']),
            w=layer['w'],
            b=layer['b']
        )

    def to_json(self):

        return {
            'type': 'FullyConnectedLayer',
            'n_in': self.n_in,
            'n_out': self.n_out,
            'activation_fn': activation_to_string(self.activation_fn),
            'w': self.w.get_value(),
            'b': self.b.get_value()
        }

    def __repr__(self):

        return json.dumps(
            {
                'Type': 'Fully-Connected, Feed-Forward Layer',
                'No. Inputs': self.n_in,
                'No. Hidden Units (Outputs)': self.n_out,
                'Activation Function': activation_to_string(self.activation_fn)
            },
            indent=4
        )

    def set_inpt(self, inpt, mini_batch_size):

        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(T.dot(self.inpt, self.w) + self.b)


class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, w=None, b=None):

        self.n_in = n_in
        self.n_out = n_out

        # Initialize weights and biases
        if w is None:
            self.w = theano.shared(np.zeros((n_in, n_out), dtype=theano.config.floatX), name='w', borrow=True)
        else:
            self.w = theano.shared(np.asarray(w, dtype=theano.config.floatX), name='w', borrow=True)

        if b is None:
            self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        else:
            self.b = theano.shared(np.asarray(b, dtype=theano.config.floatX), name='b', borrow=True)

        self.params = [self.w, self.b]

    @staticmethod
    def from_json(layer):

        return SoftmaxLayer(
            n_in=layer['n_in'],
            n_out=layer['n_out'],
            w=layer['w'],
            b=layer['b']
        )

    def to_json(self):

        return {
            'type': 'SoftmaxLayer',
            'n_in': self.n_in,
            'n_out': self.n_out,
            'w': self.w.get_value(),
            'b': self.b.get_value()
        }

    def __repr__(self):

        return json.dumps(
            {
                'Type': 'Softmax Layer',
                'No. Inputs': self.n_in,
                'No. Outputs': self.n_out
            },
            indent=4
        )

    def set_inpt(self, inpt, mini_batch_size):

        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax(T.dot(self.inpt, self.w) + self.b)

        # Need to define final output as this layer is going to be used as final layer in network
        self.y_out = T.argmax(self.output, axis=1)

    def cost(self, net):

        return -T.mean(T.log(self.output)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):

        return T.mean(T.eq(y, self.y_out))


class ExamplesIterator(object):
    """
    Used to iterate over a single file of sentence data, either for training, testing, or validating purposes.
    This means we will need to pre-compute our data split before-hand, e.g. 80:10:10 split,
    which the utility function `utils.split_data` can handily do for us.

    When iterating over the class object, we yield (input, expected output) pairs,
    where output is the index in vocabulary for target word,
    and input is a concatenated vector of one-hot vectors for the corresponding context words.
    """

    def __init__(self, file, vocab, mini_batch_size, context_size=4):
        """
        `file` points to either a training file, test file, or validation file,
        which is a big list of lines of text we use to generate supervised examples over (disjoint from one another).

        `vocab` is an initialized Vocabulary object, whose file domain is across all three files (test, train, validate)

        `mini_batch_size` is how many examples to generate at a time before yielding them together in tensor form.
        """

        self.file = file
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.context_size = context_size
        self.mini_batch_size = mini_batch_size

        # Set up an iterator to yield (target, context) pairs which we can convert into training tensors.
        # We are using this class for predictive language modelling here, so no words after target word, only before.
        self.context_pairs = MyTargetContextPairs(
            files=[self.file],
            vocab=vocab,
            before_window=self.context_size,
            after_window=0
        )

    def __iter__(self):
        """
        We yield (input, expected output) pairs for iterative NN training.
        Each yield returns `self.mini_batch_size` number of examples in a matrix.
        """

        # Running list to accumulate examples before yielding, until we have enough to constitute a mini-batch size.
        # Assert they always have the same size/length
        examples_x, examples_y = [], []

        for target, context in self.context_pairs:

            if len(context) != self.context_size:
                # We require a full number previous context words for a training example
                continue

            if len(examples_x) == self.mini_batch_size:
                # If we've accumulated enough examples for a mini-batch, yield and reset our examples store.
                yield examples_x, examples_y
                examples_x, examples_y = [], []

            # Set up a one-hot vector for each word in our previous context
            input = [np.zeros(self.vocab_size) for i in range(self.context_size)]
            for i, wd in enumerate(context):
                input[i][self.vocab[wd]] = 1.0

            # Concatenate input into one long input vector
            input = np.concatenate(input)

            output = self.vocab[target]

            examples_x.append(input)
            examples_y.append(output)

        # If we've reached end of target-context pairs iterator, and have enough examples to constitute a mini-batch,
        # then yield, otherwise we ignore those remaining examples.
        # This means we do truncate any examples at the end of our data set not an equal divisor of mini-batch size.
        # This shouldn't be too problematic if mini-batch size is small enough.
        if len(examples_x) == self.mini_batch_size:
            yield examples_x, examples_y


parser = argparse.ArgumentParser(description='Script for training a FF-NN-LM over Harry Potter text.')
parser.add_argument('--epochs', default=10, type=int, help='How many epochs to train for.')
parser.add_argument('--regularization', default=0.0, type=float, help='L2 regularization parameter.')
parser.add_argument('--context_size', default=4, type=int, help='How many previous tokens to use in predicting target.')
parser.add_argument('--min_count', default=25, type=int, help='Minimum count of frequency for valid tokens.')
parser.add_argument('--hidden_units', default=50, type=int, help='How many units in hidden layer.')
parser.add_argument('--mini_batch_size', default=64, type=int, help='How many examples in each mini-batch.')
parser.add_argument('--dimensions', default=100, type=int, help='Size of resulting word embeddings.')
parser.add_argument('--save', required=True, type=str, help='Path to save resulting model.')


if __name__ == '__main__':

    args = parser.parse_args()

    potter_files = ['data/%s.txt' % str(i) for i in range(1, 8)]

    # Split total data set into training, testing, and validation data, and obtain paths to those files
    train_data_file, test_data_file, validate_data_file = split_data(potter_files, path_to_save_dir='data_split')

    # Generate an overall vocabulary for all the testing, training, and validation data
    total_vocab = Vocabulary(files=[train_data_file, test_data_file, validate_data_file], min_count=args.min_count)

    # Initialize network and corresponding computation graph
    network = Network(
        layers=[
            EmbeddingLayer(
                dim_size=args.dimensions,
                vocab_size=len(total_vocab),
                context_size=args.context_size,
                learn_embeddings=True
            ),
            FullyConnectedLayer(
                n_in=args.dimensions * args.context_size,
                n_out=args.hidden_units,
                activation_fn=ReLU
            ),
            SoftmaxLayer(
                n_in=args.hidden_units,
                n_out=len(total_vocab)
            )
        ],
        vocab=total_vocab,
        mini_batch_size=args.mini_batch_size
    )

    # Generate our (input, expected output) iterators for training, testing, and validating
    train_iterator = ExamplesIterator(train_data_file, total_vocab, args.mini_batch_size, args.context_size)
    test_iterator = ExamplesIterator(test_data_file, total_vocab, args.mini_batch_size, args.context_size)
    validate_iterator = ExamplesIterator(validate_data_file, total_vocab, args.mini_batch_size, args.context_size)

    network.fit_iterator(
        train_iterator=train_iterator,
        test_iterator=test_iterator,
        validate_iterator=validate_iterator,
        epochs=args.epochs,
        lmbda=args.regularization,
        update_method=Network.adam
    )

    network.save(args.save)
