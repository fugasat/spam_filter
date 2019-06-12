from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Dropout, Lambda, Reshape

from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.pooling import AveragePooling1D
from keras.models import Model
from keras.models import Sequential


def create(vocab_size, embedding_dim, seq_length, lstm_units):
    """
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    input = Input(shape=(seq_length,))
    out = embedding(input)
    out = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')(out)
    out = Dropout(0.2)(out)
    #h = AveragePooling1D(pool_size=seq_length, strides=1)(h)
    out = Reshape((seq_length * lstm_units * 2,))(out)
    output = Dense(2, activation='softmax')(out)

    model = Model(inputs=input, outputs=output)
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy")
    #model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(Bidirectional(LSTM(lstm_units)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model
