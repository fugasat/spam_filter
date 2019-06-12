import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import csv
import datetime
from src import util, model


if __name__ == '__main__':
    train_file_filter = "train_00*.txt"
    test_file_filter = "test_*.txt"
    embedding_dim = 100  # 単語ベクトルの次元数
    lstm_units = 200  # LSTMの隠れ状態ベクトルの次元数
    epoch_size = 1
    batch_size = 50

    # read source data
    label_dict = util.create_label('./train_master.tsv')
    sentences_train, labels, train_files = util.read_data("./train/", train_file_filter, label_dict)
    sentences_test, labels_test, test_files = util.read_data("./test/", test_file_filter, None)

    train, test, seq_length, vocab_size = util.create_train_data(sentences_train, sentences_test)

    # create model
    model = model.create(
        vocab_size=vocab_size, embedding_dim=embedding_dim, seq_length=seq_length, lstm_units=lstm_units)

    # train param
    model.fit(train, labels, epochs=epoch_size, batch_size=batch_size)

    print("predict test data...")
    results = model.predict(test)
    model.save('model.h5', include_optimizer=False)

    result_submit = util.to_result_submit(results, test_files)
    print(result_submit)

    now = datetime.datetime.now()
    with open("./results/result_submit_{0:%Y%m%d_%H}.csv".format(now), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(result_submit)
