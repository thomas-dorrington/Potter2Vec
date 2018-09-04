from numpy import array
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, LSTM, Embedding
from utils import MyTargetContextPairs, Vocabulary


class TrainingExamplesIterator(object):

    def __init__(self, iterator, vocab):

        self.vocab = vocab
        self.iterator = iterator
        self.vocab_size = len(self.vocab)

    def __iter__(self):

        for (target, context) in self.iterator:

            X = to_categorical(y=self.vocab[target], num_classes=self.vocab_size)


if __name__ == '__main__':

    files = ["data/%s.txt" % str(i) for i in range(1, 8)]

    vocab = Vocabulary(files, min_count=25)

    pair_iterator = MyTargetContextPairs(
        files=files,
        min_count=25,
        before_window=4,
        after_window=0
    )

    examples_iterator = TrainingExamplesIterator(
        iterator=pair_iterator,
        vocab=vocab
    )

    model = Sequential()
    model.add(Embedding(len(vocab), 100, input_length=4))
    model.add(LSTM(50))
    model.add(Dense(len(vocab), activation='softmax'))
    print(model.summary())

    # compile network
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # fit network
    model.fit_generator(
        generator=examples_iterator
    )
