import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import csv
import datetime
import random
from sklearn.model_selection import train_test_split
from src import util, model
from keras.callbacks import Callback


# callback
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        print(logs)
        self.losses.append(logs.get('loss'))


if __name__ == '__main__':
    train_file_filter = "train_*.txt"
    test_file_filter = "test_*.txt"
    seq_length = 150
    embedding_dim = 50  # 単語ベクトルの次元数
    lstm_units = 50  # LSTMの隠れ状態ベクトルの次元数
    epoch_size = 5
    batch_size = 20

    # read source data
    label_dict = util.create_label('./train_master.tsv')
    sentences_train, labels, train_files = util.read_data("./train/", train_file_filter, label_dict)
    sentences_test, labels_test, test_files = util.read_data("./test/", test_file_filter, None)

    train, test, seq_length, vocab_size = util.create_train_data(sentences_train, sentences_test, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(train,
                                                        labels,
                                                        test_size=0.15,
                                                        random_state=random.randint(0, 100))

    # create model
    model = model.create_bidirectional(
        vocab_size=vocab_size, embedding_dim=embedding_dim, seq_length=seq_length, lstm_units=lstm_units)
    #model = model.create_3layer(seq_length=seq_length)

    # train param
    print(train)
    print(labels)
    losshist = LossHistory()
    model.fit(X_train, y_train, verbose=2,
              batch_size=batch_size,
              epochs=epoch_size,
              validation_data=[X_test, y_test],
              callbacks=[losshist])

    print("predict test data...")
    results = model.predict(test)
    print(results)
    model.save('model.h5', include_optimizer=False)

    result_submit = util.to_result_submit(results, test_files)
    #print(result_submit)

    now = datetime.datetime.now()
    with open("./results/result_submit_{0:%Y%m%d_%H}.csv".format(now), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(result_submit)
