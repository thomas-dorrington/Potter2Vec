import os
import theano
import numpy as np
import theano.tensor as T
from theano.tensor.nnet import softmax
from utils import MyNGrams, MySentences, Vocabulary


def ReLU(z):

    return T.maximum(0.0, z)


class Network(object):

    def __init__(self, layers, mini_batch_size):

        self.layers = layers
        self.mini_batch_size = mini_batch_size

        self.params = [param for layer in self.layers for param in layer.params]

        self.x = T.matrix("x")
        self.y = T.ivector("y")

        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)

        for j in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output, self.mini_batch_size)

        self.output = self.layers[-1].output

    def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data, lmbda=0.0):

        test_x, test_y = test_data
        training_x, training_y = training_data
        validation_x, validation_y = validation_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad) for param, grad in zip(self.params, grads)]

        i = T.lscalar()

        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            }
        )

        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            }
        )

        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            }
        )

        best_validation_accuracy = 0.0
        for epoch in xrange(epochs):

            for minibatch_index in xrange(num_training_batches):

                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))

                train_mb(minibatch_index)

                if (iteration+1) % num_training_batches == 0:

                    validation_accuracy = np.mean([validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(epoch, validation_accuracy))

                    if validation_accuracy >= best_validation_accuracy:

                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration

                        if test_data:
                            test_accuracy = np.mean([test_mb_accuracy(j) for j in xrange(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(test_accuracy))

        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration)
        )
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))


class ProjectionLayer(object):

    def __init__(self, dim_size, vocab_size, context_size):

        self.dim_size = dim_size
        self.vocab_size = vocab_size
        self.context_size = context_size

        # Initialize weights (i.e. embedding matrix)
        self.e = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=np.sqrt(1.0 / dim_size), size=(vocab_size, dim_size)),
                       dtype=theano.config.floatX),
            name='e',
            borrow=True
        )

        self.params = [self.e]

    def set_inpt(self, inpt, mini_batch_size):

        self.inpt = inpt.reshape((mini_batch_size, self.context_size*self.vocab_size))
        self.output = T.concatenate(
            [T.dot(self.inpt[:][i*self.vocab_size:(i+1)*self.vocab_size], self.e) for i in range(self.context_size)]
        )


class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=ReLU):

        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn

        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)), dtype=theano.config.floatX),
            name='w',
            borrow=True
        )

        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )

        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):

        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)


class SoftmaxLayer(object):

    def __init__(self, n_in, n_out):

        self.n_in = n_in
        self.n_out = n_out

        # Initialize weights and biases
        self.w = theano.shared(np.zeros((n_in, n_out), dtype=theano.config.floatX), name='w', borrow=True)
        self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX),name='b', borrow=True)

        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):

        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)

    def cost(self, net):
        return -T.mean(T.log(self.output)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        return T.mean(T.eq(y, self.y_out))


def size(data):

    return data[0].get_value(borrow=True).shape[0]


if __name__ == '__main__':

    GPU = True
    if GPU:

        print "Trying to run under a GPU."
        try:
            theano.config.device = 'gpu'
        except:
            # it's already set
            pass
        theano.config.floatX = 'float32'

    potter_files = [os.path.join('data/', x) for x in os.listdir('data/')]

    sentences = MySentences(
        txt_files=potter_files,
        pos_tag=False,
        verbose=False
    )

    vocab = Vocabulary(sentences)

