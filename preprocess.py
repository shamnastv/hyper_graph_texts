from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from utils import clean_str, show_statisctic, clean_document, clean_str_simple_version
import collections
from collections import Counter
import random
import numpy as np
import pickle
import json
from nltk import tokenize
from sklearn.utils import class_weight


def read_file(dataset, LDA=True):
    doc_content_list = []
    doc_sentence_list = []
    f = open('data/' + dataset + '_corpus.txt', 'rb')

    for line in f.readlines():
        doc_content_list.append(line.strip().decode('latin1'))
        doc_sentence_list.append(tokenize.sent_tokenize(clean_str_simple_version(doc_content_list[-1], dataset)))
    f.close()

    doc_content_list = clean_document(doc_sentence_list, dataset)

    max_num_sentence = show_statisctic(doc_content_list)

    doc_train_list_original = []
    doc_test_list_original = []
    labels_dic = {}
    label_count = Counter()

    i = 0
    f = open('data/' + dataset + '_labels.txt', 'r')
    lines = f.readlines()
    for line in lines:
        temp = line.strip().split("\t")
        if temp[1].find('test') != -1:
            doc_test_list_original.append((doc_content_list[i], temp[2]))
        elif temp[1].find('train') != -1:
            doc_train_list_original.append((doc_content_list[i], temp[2]))
        if not temp[2] in labels_dic:
            labels_dic[temp[2]] = len(labels_dic)
        label_count[temp[2]] += 1
        i += 1

    f.close()
    print(label_count)

    word_freq = Counter()
    word_set = set()
    for doc_words in doc_content_list:
        for words in doc_words:
            for word in words:
                word_set.add(word)
                word_freq[word] += 1

    vocab = ['<pad>'] + list(word_set)
    vocab_size = len(vocab)

    vocab_dic = {}
    for i, word in enumerate(word_set):
        vocab_dic[word] = i

    print('Total_number_of_words: ' + str(len(vocab)))
    print('Total_number_of_categories: ' + str(len(labels_dic)))

    doc_train_list = []
    doc_test_list = []

    for doc, label in doc_train_list_original:
        temp_doc = []
        for sentence in doc:
            temp = []
            for word in sentence:
                temp.append(vocab_dic[word])
            temp_doc.append(temp)
        doc_train_list.append((temp_doc, labels_dic[label]))

    for doc, label in doc_test_list_original:
        temp_doc = []
        for sentence in doc:
            temp = []
            for word in sentence:
                temp.append(vocab_dic[word])
            temp_doc.append(temp)
        doc_test_list.append((temp_doc, labels_dic[label]))

    # keywords_dic = {}
    # if LDA:
    #     keywords_dic_original = pickle.load(open('data/' + dataset + '_LDA.p', "rb"))
    #
    #     for i in keywords_dic_original:
    #         if i in vocab_dic:
    #             keywords_dic[vocab_dic[i]] = keywords_dic_original[i]

    train_set_y = [j for i, j in doc_train_list]

    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(train_set_y), y=train_set_y)

    return doc_content_list, doc_train_list, doc_test_list, vocab_dic, labels_dic, class_weights


def get_embedding(word_to_id):
    filename = '../glove.840B.300d.txt'
    dim = 300

    embeddings = np.random.uniform(-0.25, 0.25, (len(word_to_id), dim))
    with open(filename, encoding='utf-8') as fp:
        for line in fp:
            elements = line.strip().split()
            word = elements[0]
            if word in word_to_id:
                try:
                    embeddings[word_to_id[word]] = [float(v) for v in elements[1:]]
                except ValueError:
                    pass

    embeddings[0] = np.zeros(dim, dtype='float32')
    return embeddings
