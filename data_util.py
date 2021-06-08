import os
import pickle
from math import log

from preprocess import read_file, get_embedding
import numpy as np

from utils import create_word_vectors


class Data:
    def __init__(self, data, keywords, lda=True):
        self.full_doc = data[2]
        self.label = data[1]
        doc = data[0]

        vocab_set = set()
        for sent in doc:
            for word in sent:
                vocab_set.add(word)

        vocab_list = list(vocab_set)
        self.node_ids = vocab_list
        wf = [0] * len(vocab_list)

        word_to_id = {}
        for i, w in enumerate(vocab_list):
            word_to_id[w] = i

        self.data = []
        for sent in doc:
            temp = set()
            for word in sent:
                temp.add(word_to_id[word])
                wf[word_to_id[word]] += 1
            temp = list(temp)
            if temp:
                self.data.append(temp)
        s = sum(wf)
        self.tf = [i/s for i in wf]

        self.rows = []
        self.cols = []
        self.vals = []

        self.degrees_e = [0] * len(self.data)
        self.degrees_v = [0] * len(self.node_ids)

        for i, sent in enumerate(self.data):
            for j in sent:
                self.degrees_e[i] += 1
                self.degrees_v[j] += 1
                self.rows.append(j)
                self.cols.append(i)
                self.vals.append(1.0)

        if len(self.cols) == 0:
            s = 0
        else:
            s = max(self.cols) + 1

        if lda:
            for w in vocab_list:
                if w in keywords:
                    temp = keywords[w]
                    j = word_to_id[w]
                    self.rows += [j] * len(temp)
                    cols = [topic + s for topic in temp]
                    self.cols += cols
                    self.vals += [1.0] * len(temp)
                    self.degrees_e += [0] * (max(cols) - len(self.degrees_e) + 1)
                    for i in cols:
                        self.degrees_e[i] += 1
                    self.degrees_v[j] += len(temp)

        self.edge_num = len(self.degrees_e)
        self.node_num = len(self.degrees_v)

        self.degrees_e = [1/i if i != 0 else 0 for i in self.degrees_e]
        self.degrees_v = [1/i if i != 0 else 0 for i in self.degrees_v]

        self.d_type = 0
        self.idf = None
        self.cluster = 0


def get_data(dataset, lda=True, val_prop=.1, seed=0):
    lda_str = ''
    if lda:
        lda_str = '_lda'
    pickle_file = './data/%s%s_dump.pkl' % (dataset, lda_str)
    if os.path.exists(pickle_file):
        data = pickle.load(open(pickle_file, 'rb'))
        train_dev_data, test_data, vocab_dic, labels_dic, class_weights, word_vectors = data
    else:
        doc_content_list, doc_train_list, doc_test_list, vocab_dic, labels_dic, class_weights, keywords_dic\
            = read_file(dataset, lda)
        # word_vectors = get_embedding(vocab_dic)
        word_vectors = create_word_vectors(doc_content_list, vocab_dic)
        # word_vectors = np.concatenate((word_vectors, word_vectors1), axis=1)
        d_f = [0] * len(word_vectors)
        d_f[0] = 1

        train_dev_data = []
        for d in doc_train_list:
            d_t = Data(d, keywords_dic, lda)
            train_dev_data.append(d_t)
            for w in d_t.node_ids:
                d_f[w] += 1

        test_data = []
        for d in doc_test_list:
            d_t = Data(d, keywords_dic, lda)
            test_data.append(d_t)
            for w in d_t.node_ids:
                d_f[w] += 1

        total_size = len(doc_train_list) + len(doc_test_list)
        d_f = [log(total_size / (1 + i)) + 1 for i in d_f]
        for d in train_dev_data:
            idf = [d_f[i] for i in d.node_ids]
            s = sum(idf)
            idf = [i/s for i in idf]
            d.idf = idf

        d_f = [log(total_size / (1 + i)) + 1 for i in d_f]
        for d in test_data:
            idf = [d_f[i] for i in d.node_ids]
            s = sum(idf)
            idf = [i/s for i in idf]
            d.idf = idf

        data = train_dev_data, test_data, vocab_dic, labels_dic, class_weights, word_vectors
        pickle.dump(data, open(pickle_file, 'wb'))

    np.random.seed(seed)
    total_size = len(train_dev_data)
    val_size = int(val_prop * total_size)
    train_size = total_size - val_size

    idx = np.random.permutation(total_size)
    train_idx = idx[:train_size]
    dev_idx = idx[train_size:]

    train_data = []
    dev_data = []

    for i in train_idx:
        train_data.append(train_dev_data[i])

    for i in dev_idx:
        dev_data.append(train_dev_data[i])

    for d in dev_data:
        d.d_type = 1

    for d in test_data:
        d.d_type = 2

    return train_data, dev_data, test_data, vocab_dic, labels_dic, class_weights, word_vectors


if __name__ == '__main__':
    # train_data_n = get_data(dataset='R8', lda=True)[0]
    # print(train_data_n[100].rows, train_data_n[100].cols)
    data_n = get_data(dataset='R8')
    for j in range(3):
        print(len(data_n[j]))
        for d_n in data_n[j]:
            print(d_n.d_type, end=' ')
        print('\n')
