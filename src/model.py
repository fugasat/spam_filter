from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Dropout, Lambda, Reshape

from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.pooling import AveragePooling1D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam


def create_bidirectional(vocab_size, embedding_dim, seq_length, lstm_units):
    """
    https://paper.hatenadiary.jp/entry/2016/10/19/231911
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length))
    model.add(Bidirectional(LSTM(lstm_units)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_3layer(seq_length):
    """
    MNISTサンプルから引用
    """
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(seq_length,)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model
