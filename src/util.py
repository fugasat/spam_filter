import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from nltk.tokenize import word_tokenize

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split


def create_label(path):
    df = pd.read_csv(path, sep="\t")
    return dict(zip(list(df.file_name), list(df.label)))


def normalization(s):
    s = s.lower()
    s = re.sub("[0-9!-@[-_{-~]", "", s)  # 数字と記号を除去
    return s


def read_data(path, file_filter, label_dict):
    sentences = []
    labels = []
    file_path = Path(path)
    data_files = list(file_path.glob(file_filter))

    for data_file in data_files:
        with open(data_file) as f:
            s = f.read()
            s = normalization(s)

            s = word_tokenize(s)
            s = s[1:]  # 先頭のsubjectを除去

            sentences.append(s)

            if label_dict:
                labels.append(label_dict[os.path.basename(data_file)])

    #if label_dict:
    #    labels = to_categorical(labels)

    return sentences, labels, data_files


def create_word2id(sentences):
    vocab = {}
    for s in sentences:
        for w in s:
            vocab[w] = vocab.get(w, 0) + 1

    word2id = {'<unk>': 0}
    for w, v in vocab.items():
        if w not in word2id and v >= 2:
            word2id[w] = len(word2id)

    vocab_size = len(word2id.keys())  # 扱う語彙の数
    return word2id, vocab_size


def create_train_data(sentences_train, sentences_test):
    word2id, vocab_size = create_word2id(sentences_train + sentences_test)
    train = [[word2id.get(w, 0) for w in s] for s in sentences_train]
    test = [[word2id.get(w, 0) for w in s] for s in sentences_test]

    target_len = []
    for t in train:
        target_len.append(len(t))
    for t in test:
        target_len.append(len(t))

    seq_length = max(target_len)  # 入力ベクトルの次元数（文章の長さ）
    train = pad_sequences(train, maxlen=seq_length, dtype=np.int32, padding='post', truncating='post', value=0)
    test = pad_sequences(test, maxlen=seq_length, dtype=np.int32, padding='post', truncating='post', value=0)
    return train, test, seq_length, vocab_size


def to_result_submit(results, test_files):
    result_submit = []
    for i, r in enumerate(results):
        result_flag = 0
        #if r[0] > r[1]:
        if r[0] > 0.5:
                result_flag = 1
        result_submit.append([os.path.basename(test_files[i]), result_flag])
    return result_submit
