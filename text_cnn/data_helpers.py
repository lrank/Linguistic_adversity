import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data(data_filename):
    text = list(open(data_filename, "r").readlines())
    text = [s.strip() for s in text]
    # print text[0]

    i = 0
    res = []
    while i < len(text):
        doc = []
        for j in range( int(text[i]) ):
            doc.append( text[i + j + 1] )
        i += int(text[i]) + 1
        
        res.append( doc )
    return res


def load_data_and_labels(positive_data_file, negative_data_file):
    # Load data from files
    positive_examples = load_data(positive_data_file)
    negative_examples = load_data(negative_data_file)
    x_text = positive_examples + negative_examples
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_test_data_and_labels(test_data_file):
    # Load data from files
    text = list(open(test_data_file, "r").readlines())
    text = [s.strip() for s in text]
    # Fetch labels
    x_text = []
    y = []
    for d in text:
        if int(d[0]) == 1:
            y.append( [0, 1] )
        else:
            y.append( [1, 0] )
        x_text.append( d[2:] )
    y = np.array( y )
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, dim = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * dim
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    for word in vocab:
        if word not in word_vecs:# and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  
