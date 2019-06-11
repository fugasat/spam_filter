from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Dropout, Lambda, Reshape

from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.pooling import AveragePooling1D
from keras.models import Model


def create(vocab_size, embedding_dim, seq_length, lstm_units):
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    input = Input(shape=(seq_length,))
    embed = embedding(input)
    bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')(embed)
    h = Dropout(0.2)(bilstm)
    #h = AveragePooling1D(pool_size=seq_length, strides=1)(h)
    h = Reshape((seq_length * lstm_units * 2,))(h)
    output = Dense(2, activation='softmax')(h)

    model = Model(inputs=input, outputs=output)
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy")
    #model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    return model
