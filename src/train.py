import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import util, model


if __name__ == '__main__':
    # read source data
    label_dict = util.create_label('./train_master.tsv')
    sentences, labels = util.read_train_file("./train/", "train_000*.txt", label_dict)

    word2id, vocab_size = util.create_word2id(sentences)
    train, seq_length = util.create_train_data(sentences, word2id)

    # create model
    embedding_dim = 100  # 単語ベクトルの次元数
    lstm_units = 200  # LSTMの隠れ状態ベクトルの次元数
    model = model.create(
        vocab_size=vocab_size, embedding_dim=embedding_dim, seq_length=seq_length, lstm_units=lstm_units)

    # train param
    epoch_size = 1
    batch_size = 50
    model.fit(train, labels, epochs=epoch_size, batch_size=batch_size)

    results = model.predict(train)
    model.save('model.h5', include_optimizer=False)
    print("predict:", results)
    print("label:", labels)

    result_submit = []
    for r in results:
        if r[0] > r[1]:
            result_submit.append(1)
        else:
            result_submit.append(0)

    print("submit:", result_submit)
