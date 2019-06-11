import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import util, model


if __name__ == '__main__':
    # read source data
    label_dict = util.create_label('./train_master.tsv')
    sentences_train, labels, train_files = util.read_data("./train/", "train_000*.txt", label_dict)
    sentences_test, labels_test, test_files = util.read_data("./test/", "test_000*.txt", None)

    train, test, seq_length, vocab_size = util.create_train_data(sentences_train, sentences_test)

    # create model
    embedding_dim = 100  # 単語ベクトルの次元数
    lstm_units = 200  # LSTMの隠れ状態ベクトルの次元数
    model = model.create(
        vocab_size=vocab_size, embedding_dim=embedding_dim, seq_length=seq_length, lstm_units=lstm_units)

    # train param
    epoch_size = 1
    batch_size = 50
    model.fit(train, labels, epochs=epoch_size, batch_size=batch_size)

    results = model.predict(test)
    model.save('model.h5', include_optimizer=False)
    #print("predict:", results)

    result_submit = util.to_result_submit(results, test_files)
    print("submit:", result_submit)
