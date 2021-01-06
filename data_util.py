from preprocess import read_file
import numpy as np


class Data:
    def __init__(self, data, vocab_dic):
        self.label = data[1]
        doc = data[0]

        vocab_set = set()
        for sent in doc:
            for word in sent:
                vocab_set.add(word)

        vocab_list = list(vocab_set)
        word_to_id = {}
        for i, w in enumerate(vocab_list):
            word_to_id[w] = i

        self.node_ids = vocab_list
        # for w in vocab_list:
        #     self.node_ids.append(vocab_dic[w])

        self.data = []
        for sent in doc:
            temp = []
            for word in sent:
                temp.append(word_to_id[word])
            if temp:
                self.data.append(temp)

        self.node_num = len(self.node_ids)
        self.edge_num = len(self.data)

        self.rows = []
        self.cols = []
        self.vals = []

        self.degrees_e = [0] * self.edge_num
        self.degrees_v = [0] * self.node_num

        for i, sent in enumerate(self.data):
            for j in sent:
                self.degrees_e[i] += 1
                self.degrees_v[j] += 1
                self.rows.append(j)
                self.cols.append(i)
                self.vals.append(1.0)

        self.degrees_e = [1/i if i != 0 else 0 for i in self.degrees_e]
        self.degrees_v = [1/i if i != 0 else 0 for i in self.degrees_v]


def get_data(dataset, vla_prop=.2):
    doc_content_list, doc_train_list, doc_test_list, vocab_dic, labels_dic, class_weights = read_file(
        dataset)

    train_dev_data = []
    for d in doc_train_list:
        train_dev_data.append(Data(d, vocab_dic))

    test_data = []
    for d in doc_test_list:
        test_data.append(Data(d, vocab_dic))

    total_size = len(train_dev_data)
    val_size = int(vla_prop * total_size)
    train_size = total_size - val_size

    idx = np.random.permutation(len(train_dev_data))
    train_idx = idx[:train_size]
    dev_idx = idx[train_size:]

    train_data = []
    dev_data = []

    for i in train_idx:
        train_data.append(train_dev_data[i])

    for i in dev_idx:
        dev_data.append(train_dev_data[i])

    return train_data, dev_data, test_data, vocab_dic, labels_dic, class_weights

