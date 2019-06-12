from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Dropout, Lambda, Reshape

from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.pooling import AveragePooling1D
from keras.models import Model
from keras.models import Sequential


def create(vocab_size, embedding_dim, seq_length, lstm_units):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(Bidirectional(LSTM(lstm_units)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model
