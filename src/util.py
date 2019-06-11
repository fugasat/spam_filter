import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from nltk.tokenize import word_tokenize

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical


def create_label(path):
    df = pd.read_table(path)
    return dict(zip(list(df.file_name), list(df.label)))


def normalization(s):
    s = s.lower()
    s = re.sub("[0-9!-@[-_{-~]", "", s)  # 数字と記号を除去
    return s


def read_train_file(path, file_filter, label_dict):
    sentences = []
    labels = []
    train_path = Path(path)
    train_files = list(train_path.glob(file_filter))

    for train_file in train_files:
        with open(train_file) as f:
            s = f.read()
            s = normalization(s)

            s = word_tokenize(s)
            s = s[1:]  # 先頭のsubjectを除去

            sentences.append(s)
            labels.append(label_dict[os.path.basename(train_file)])

    labels = to_categorical(labels)

    return sentences, labels


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


def create_train_data(sentences, word2id):
    train = [[word2id.get(w, 0) for w in s] for s in sentences]
    target_len = []
    for t in train:
        target_len.append(len(t))

    seq_length = max(target_len)  # 入力ベクトルの次元数（文章の長さ）
    train = pad_sequences(train, maxlen=seq_length, dtype=np.int32, padding='post', truncating='post', value=0)
    return train, seq_length
