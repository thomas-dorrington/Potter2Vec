import os
import json
import theano
import argparse
import numpy as np
import cPickle as pickle
import theano.tensor as T
from theano.printing import pydotprint
from theano.tensor.nnet import softmax
from preprocessors import PreprocessorV1
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
     Allows us to train such a network in a supervised fashion, with methods for loading and saving models to disk.
     """

    def __init__(self, layers, vocab, mini_batch_size, statistics=None):
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

        # We store one training loss per iteration, the average of the cost of each example in that mini-batch;
        # one test loss per epoch, the average of the cost of all the test mini-batches;
        # and one validate loss per epoch, the average of the cost of all the validate mini-batches.
        # Similarly for accuracy
        if statistics is None:
            self.statistics = {
                'costs': {'train': [], 'test': [], 'validate': []},
                'accuracies': {'train': [], 'test': [], 'validate': []}
            }
        else:
            self.statistics = statistics

        # Extract all the parameters from all the layers in the network that we want to learn
        self.params = [param for layer in self.layers for param in layer.params]

        # Symbolically set up networks input and output
        self.x = T.matrix("x")
        self.y = T.ivector("y")

        # Set the first layers input to our symbolic network input `self.x`
        init_layer = self.layers[0]
        init_layer.set_inpt(inpt=self.x, mini_batch_size=self.mini_batch_size)

        # Propagate each layers' output to the next layers input
        for j in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(inpt=prev_layer.output, mini_batch_size=self.mini_batch_size)

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
            mini_batch_size=mini_batch_size if mini_batch_size is not None else model['mini_batch_size'],
            statistics=model['statistics']
        )

    def save(self, path_to_model):
        """
        Compiles model into JSON format, than saves as a pickle dump.
        """

        model = {
            'layers': [layer.to_json() for layer in self.layers],
            'vocab': self.vocab.to_json(),
            'mini_batch_size': self.mini_batch_size,
            'statistics': self.statistics
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

    def predict_example(self, x):
        """
        Takes input `x` to network, a concatenated vector of one-hot vectors representing context words.
        Returns probability distribution over vocabulary for predicted target words.
        We are assuming `x` is a single input, not a mini-batch matrix of multiple inputs.
        We need to wrap `x` in a  one element sized array, because network expects mini-batches, i.e. arrays.
        Similarly, need to get the first element of output matrix.
        """

        # This method will only work at predicting one example at a time
        assert self.mini_batch_size == 1

        predict = theano.function(
            inputs=[],
            outputs=self.output,
            givens={self.x: np.asarray([x])}
        )

        return predict()[0]

    @staticmethod
    def adam(cost, params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=1 - 1e-8):
        """
        Adam update rules for learning; improves over traditional SGD.
        Credit: https://gist.github.com/skaae/ae7225263ca8806868cb
        Based on: http://arxiv.org/pdf/1412.6980v4.pdf
        """

        grads = theano.grad(cost, params)   # Get gradients wrt. stochastic objective function
        t = theano.shared(np.float32(1.0))  # Initialize time-step to 1
        b1_t = b1 * gamma ** (t - 1.0)      # Decay the first moment running average coefficient

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
        Simply move the parameters a small direction in the negative of gradient of cost function
        """

        grads = T.grad(cost, params)
        updates = [(param, param - eta * grad) for param, grad in zip(params, grads)]

        return updates

    def fit_iterator(self,
                     train_iterator,
                     test_iterator,
                     validate_iterator,
                     epochs,
                     lmbda=0.0,
                     update_method=adam,
                     path_to_save=None):
        """
        Train `self` in iterator-like fashion.
        `train_iterator`, `test_iterator`, and `validate_iterator`, are all iterators that yield
        `self.mini_batch_size` number of (input, expected output) example pairs in tensor format.
        Generating examples on the fly is much more efficient than loading everything into memory at once.

        If `path_to_save` is not None, we want to save the model at the end of each epoch in this directory.
        """

        # If we want to save the model at end of each epoch, check the target directory exists
        if path_to_save is not None and not os.path.exists(path_to_save):
            os.mkdir(path_to_save)

        # Define the (L2 regularized) cost function
        # We do not divide L2 regularization term by number of training batches; `lmbda` should be adjusted accordingly
        l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + 0.5 * lmbda * l2_norm_squared

        # Find symbolic update rules
        updates = update_method(cost=cost, params=self.params)

        # Set up symbolic variables to hold mini-batch inputs & outputs
        x = T.matrix("x")
        y = T.ivector("y")

        # For a given mini-batch of inputs (`x`) and outputs (`y`), compute the loss, and update parameters accordingly
        train_mb = theano.function(
            inputs=[x, y],
            outputs={'accuracy': self.layers[-1].accuracy(self.y), 'cost': cost},
            updates=updates,
            givens={self.x: x, self.y: y}
        )

        # Accuracy and cost for a given mini-batch of inputs (`x`) and outputs (`y`)
        mb_statistics = theano.function(
            inputs=[x, y],
            outputs={'accuracy': self.layers[-1].accuracy(self.y), 'cost': cost},
            givens={self.x: x, self.y: y}

        )

        # Do the actual training
        for epoch in xrange(epochs):

            # Keep track of how many mini-batches we've trained for in this epoch for progress logging
            mini_batch_index = 0

            for train_x, train_y in train_iterator:
                # Each iteration of iterator yields one mini-batch

                if mini_batch_index % 100 == 0:
                    print("Training mini-batch number {0} of epoch {1}".format(mini_batch_index, epoch))

                train_stats = train_mb(train_x, train_y)

                self.statistics['costs']['train'].append(train_stats['cost'])
                self.statistics['accuracies']['train'].append(train_stats['accuracy'])

                mini_batch_index += 1  # Increment mini-batch counter

            print("Finished training epoch {0}, with {1} mini-batches\n".format(epoch, mini_batch_index))

            print("Calculating validation scores and costs ...\n")

            # At end of epoch, calculate the average validation mini-batch cost and accuracy
            validate_accuracy, validate_cost = [], []
            for validate_x, validate_y in validate_iterator:

                validate_stats = mb_statistics(validate_x, validate_y)
                validate_cost.append(validate_stats['cost'])
                validate_accuracy.append(validate_stats['accuracy'])

            # Average validate cost and accuracy across all mini-batches
            validate_cost = np.mean(validate_cost)
            validate_accuracy = np.mean(validate_accuracy)

            print("Calculating test scores and costs ...\n")

            # At end of epoch, also calculate average test mini-batch cost and accuracy
            test_accuracy, test_cost = [], []
            for test_x, test_y in test_iterator:

                test_stats = mb_statistics(test_x, test_y)
                test_cost.append(test_stats['cost'])
                test_accuracy.append(test_stats['accuracy'])

            # Average test cost and accuracy across all mini-batches
            test_cost = np.mean(test_cost)
            test_accuracy = np.mean(test_accuracy)

            print("End of epoch {0}: validation accuracy of {1:.2%} and cost of {2:.2}\n".format(
                epoch, validate_accuracy, validate_cost)
            )

            print("End of epoch {0}: test accuracy of {1:.2%} and cost of {2:.2}\n".format(
                epoch, test_accuracy, test_cost)
            )

            # Store statistics results in accumulating attribute

            self.statistics['costs']['test'].append(test_cost)
            self.statistics['costs']['validate'].append(validate_cost)

            self.statistics['accuracies']['test'].append(test_accuracy)
            self.statistics['accuracies']['validate'].append(validate_accuracy)

            # Potentially save model at end of each iteration
            if path_to_save is not None:
                self.save(os.path.join(path_to_save, 'iter_%s.bin' % str(epoch)))

        print("Finished training network.")


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

        # The number of total input connections this layer (not to the embedding matrix) is the product of:
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

        self.y_out = T.argmax(self.output, axis=1)  # Need a final predicted output for this layer

    def cost(self, net):
        """
        Calculates log-likelihood cost function. A lot going on here ...

        `net.y` will be set to a vector of integers, one integer for each example in the current mini-batch.
        Each integer then corresponds to the index in the vocabulary we are expecting for the target word.

        `T.arange(net.y.shape[0])` returns evenly spaced integers in range [0, number of examples in mini-batch]

        `T.log` just logs values every element in `self.output` matrix.

        By indexing into `T.log(self.output)` with indices `[T.arange(net.y.shape[0]), net.y]`,
        we are picking out, for each example in the mini-batch, the index in the predicted vector of probabilities,
        which corresponds to the index it should be 1 in according to our expected `net.y` vector.
        We are doing a hard classification task, so only one index in the predicted ouput vector of probabilities
        should be set to 1.

        We then average these probabilities for our final cost.
        If we did a good job of prediction, the probabilities in the expected word indices should be close to 1.0,
        and the other probabilities across the rest of the predicted probability distribution all close to 0.0
        """

        return -T.mean(T.log(self.output)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):

        return T.mean(T.eq(y, self.y_out))


class ExamplesIterator(object):
    """
    Used to iterate over a single file of sentence data, either for training, testing, or validating purposes.
    This means we will need to pre-compute our data split before-hand, e.g. 80:10:10 split,
    which the utility function `utils.split_data` can handily do for us.

    When iterating over an instance of this class object, we yield mini-batch number of (input, expected output) pairs,
    where output is the integer index in vocabulary for target word,
    and input is a concatenated vector of one-hot vectors for the corresponding context words.
    """

    def __init__(self,
                 file,
                 vocab,
                 mini_batch_size,
                 context_size=4,
                 preprocessor=None):
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
            vocab=self.vocab,
            before_window=self.context_size,
            after_window=0,
            preprocessor=preprocessor
        )

    def __iter__(self):
        """
        We yield (input, expected output) pairs for iterative NN training.
        Each yield returns `self.mini_batch_size` number of examples.
        """

        # Running list to accumulate examples, until enough to constitute a mini-batch size, at which point we yield
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

            # Concatenate input into one long input vector, then back from np.array to Python list
            # Lists are easier to pass around, especially if loading & saving using OptimizedExamplesIterator
            input = np.concatenate(input).tolist()

            # Get integer index of target word
            output = self.vocab[target]

            examples_x.append(input)
            examples_y.append(output)

        # If we've reached end of target-context pairs iterator, and have enough examples to constitute a mini-batch,
        # then yield, otherwise we ignore those remaining examples.
        # This means we do truncate any examples at the end of our data set not an equal divisor of mini-batch size.
        # This shouldn't be too problematic if mini-batch size is small enough.
        if len(examples_x) == self.mini_batch_size:
            yield examples_x, examples_y


class OptimizedExamplesIterator(object):
    """
    We need to significantly speed up the loading and pre-processing of data to yield for inference and training.
    Therefore, we pre-generate all our mini-batches of examples, and save to disk.
    So, yielding examples for training solely consists of loading files one at a time,
    and iterating over each mini-batch in the file, yielding one at a time.
    """

    def __init__(self,
                 file,
                 vocab,
                 mini_batch_size,
                 dir_to_save,
                 context_size=4,
                 preprocessor=None,
                 mini_batches_per_file=100):
        """
        All arguments the same as ordinary ExamplesIterator, except:

        - `dir_to_save` is the path to the directory we are going to save all the files of pre-computed mini-batches.
           If this already exists, we assume it holds pre-computed mini-batches, and we do not need to generate any more

        - `mini_batches_per_file` is how many mini-batches to save in each file.
        """

        self.file = file
        self.vocab = vocab
        self.dir_to_save = dir_to_save
        self.context_size = context_size
        self.vocab_size = len(self.vocab)
        self.preprocessor = preprocessor
        self.mini_batch_size = mini_batch_size
        self.mini_batches_per_file = mini_batches_per_file

        if os.path.exists(self.dir_to_save):
            # We have already pre-computed and saved our mini-batches to disk.
            # Load these to use for training.
            self.paths_to_files = [os.path.join(self.dir_to_save, x) for x in os.listdir(self.dir_to_save)]

        else:
            # We are computing our mini-batches to disk for the first time
            os.mkdir(self.dir_to_save)
            self.paths_to_files = self._init_data()

    def __iter__(self):
        """
        Iterate through saved files; in each file, iterate through saved off mini-batches
        """

        for f in self.paths_to_files:

            with open(f, 'r') as open_f:
                examples = json.load(open_f)

            for x, y in examples:
                # Each `x` is a mini-batch of examples, e.g. a list of 64 or 128 examples
                # We saved the concatenated one-hot vectors off to disk compressed, storing only the indices of ones
                # We need to convert them back here to those sparse vectors

                for x_index in xrange(len(x)):

                    one_hot_x = np.zeros(self.context_size * self.vocab_size)
                    for i in x[x_index]:
                        one_hot_x[i] = 1.0

                    x[x_index] = one_hot_x

                yield x, y

    def _init_data(self):

        print("Pre-computing mini-batches and saving to directory: %s.\n" % self.dir_to_save)

        iterator = ExamplesIterator(
            file=self.file,
            vocab=self.vocab,
            mini_batch_size=self.mini_batch_size,
            context_size=self.context_size,
            preprocessor=self.preprocessor
        )

        # Accumulate list of all paths to files we have saved mini-batches under, under this model
        paths_to_files = []

        # Keep accumulating pairs of (input, expected output) mini-batch pairs in `examples`,
        # until we have `mini_batches_per_file`, at which point we save them to disk and reset our examples store.
        # `file_index` is the number of files we've iterated through, used to know what to call the next file.
        examples = []
        file_index = 0

        for x, y in iterator:
            # Each `x`, `y` is mini-batch number of inputs and expected output pairs.
            # That is, `len(x) == len(y) == self.mini_batch_size`

            if len(examples) == self.mini_batches_per_file:

                path_to_file = os.path.join(self.dir_to_save, '%s.bin' % str(file_index))
                paths_to_files.append(path_to_file)

                with open(path_to_file, 'w') as open_f:
                    json.dump(examples, open_f)

                examples = []    # Reset examples store
                file_index += 1  # Increment number of files written to

                print("Written %s files to disk so far, each storing %s mini-batches of examples" %
                      (str(file_index), str(self.mini_batches_per_file)))

            # We compress input of concatenated one-hot vectors to drastically save time loading & saving
            # by storing only the indices of the 1.0's, not all the zeros
            for x_index in xrange(len(x)):
                # Iterate through each example in the mini-batch `x`
                # Get all the indices at which the example is 1.0, and reassign it

                x[x_index] = [i for i, el in enumerate(x[x_index]) if el == 1.0]

            examples.append((x, y))

        if len(examples) > 0:
            # Dump last remaining examples that are not an even divisor of `self.mini_batches_per_file`

            path_to_file = os.path.join(self.dir_to_save, '%s.bin' % str(file_index))
            paths_to_files.append(path_to_file)

            with open(path_to_file, 'w') as open_f:
                json.dump(examples, open_f)

            print("Written %s files to disk, this last one storing %s mini-batches of examples" %
                  (str(file_index), str(len(examples))))

        print("Finished saving pre-computing mini-batches to disk across {0} files\n".format(len(paths_to_files)))

        return paths_to_files


parser = argparse.ArgumentParser(description='Script for training a FF-NN-LM over Harry Potter text.')
parser.add_argument('--epochs', default=10, type=int, help='How many epochs to train for.')
parser.add_argument('--regularization', default=0.0, type=float, help='L2 regularization parameter.')
parser.add_argument('--context_size', default=4, type=int, help='How many previous tokens to use in predicting target.')
parser.add_argument('--min_count', default=25, type=int, help='Minimum count of frequency for valid tokens.')
parser.add_argument('--hidden_units', default=50, type=int, help='How many units in hidden layer.')
parser.add_argument('--mini_batch_size', default=64, type=int, help='How many examples in each mini-batch.')
parser.add_argument('--dimensions', default=100, type=int, help='Size of resulting word embeddings.')
parser.add_argument('--save', required=True, type=str, help='Path to directory save resulting models.')


if __name__ == '__main__':

    args = parser.parse_args()

    potter_files = ['data/%s.txt' % str(i) for i in range(1, 8)]

    # Split total data set into training, testing, and validation data, and obtain paths to those files
    train_data_file, test_data_file, validate_data_file = split_data(files=potter_files, path_to_save_dir='data_split')

    # Initialize a token pre-processor object to share across classes
    preprocessor = PreprocessorV1()

    # Generate an overall vocabulary for all the testing, training, and validation data
    total_vocab = Vocabulary(
        files=[train_data_file, test_data_file, validate_data_file],
        min_count=args.min_count,
        preprocessor=preprocessor
    )

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

    train_iterator = OptimizedExamplesIterator(
        file=train_data_file,
        vocab=total_vocab,
        mini_batch_size=args.mini_batch_size,
        dir_to_save=os.path.join('data_split', 'train_mini_batches'),
        context_size=args.context_size,
        preprocessor=preprocessor,
        mini_batches_per_file=250
    )

    test_iterator = OptimizedExamplesIterator(
        file=test_data_file,
        vocab=total_vocab,
        mini_batch_size=args.mini_batch_size,
        dir_to_save=os.path.join('data_split', 'test_mini_batches'),
        context_size=args.context_size,
        preprocessor=preprocessor,
        mini_batches_per_file=250
    )

    validate_iterator = OptimizedExamplesIterator(
        file=validate_data_file,
        vocab=total_vocab,
        mini_batch_size=args.mini_batch_size,
        dir_to_save=os.path.join('data_split', 'validate_mini_batches'),
        context_size=args.context_size,
        preprocessor=preprocessor,
        mini_batches_per_file=250
    )

    # Train the network
    network.fit_iterator(
        train_iterator=train_iterator,
        test_iterator=test_iterator,
        validate_iterator=validate_iterator,
        epochs=args.epochs,
        lmbda=args.regularization,
        update_method=lambda cost, params: Network.adam(cost, params, learning_rate=0.0005),
        path_to_save=args.save
    )
