import json

import numpy
from collections import Counter
import pickle
import platform

def load_data():
    test = [[],[]]
    train = _parse_data(open('demo.data', 'rb'))
    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
   
    vocab.insert(0, '')
    chunk_tags = numpy.unique(numpy.array([row[1] for sample in train for row in sample if len(row) == 2]))

    train = _process_data(train, vocab, chunk_tags)
    return train, test, (vocab, chunk_tags)

def _parse_data(fh):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
    #  you have to use recorsponding instructions

    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'

    string = fh.read().decode('utf-8')
    data = [[row.split(", ") for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text)]
    fh.close()
    return data

def pad_sequences(arrays, maxlen, value = 0):
    for array in arrays:
        while(len(array) != maxlen):
            array.append(value)
    return arrays

def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab)) # 製作 word 2 id array
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    y_chunk = [[list(chunk_tags).index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, maxlen)  # left padding
    y_chunk = pad_sequences(y_chunk, maxlen, value = -1)

    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]

    return x, y_chunk
