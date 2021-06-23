from math import log

import scipy.sparse as sp
from nltk.corpus import stopwords

from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.utils import *
from sklearn.decomposition import TruncatedSVD


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
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


def clean_str_simple_version(string, dataset):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def show_statisctic(clean_docs):
    min_len = 10000
    aver_len = 0
    max_len = 0
    num_sentence = sum([len(i) for i in clean_docs])
    ave_num_sentence = num_sentence * 1.0 / len(clean_docs)

    for doc in clean_docs:
        for sentence in doc:
            temp = sentence
            aver_len = aver_len + len(temp)

            if len(temp) < min_len:
                min_len = len(temp)
            if len(temp) > max_len:
                max_len = len(temp)

    aver_len = 1.0 * aver_len / num_sentence

    print('min_len_of_sentence : ' + str(min_len))
    print('max_len_of_sentence : ' + str(max_len))
    print('min_num_of_sentence : ' + str(min([len(i) for i in clean_docs])))
    print('max_num_of_sentence : ' + str(max([len(i) for i in clean_docs])))
    print('average_len_of_sentence: ' + str(aver_len))
    print('average_num_of_sentence: ' + str(ave_num_sentence))
    print('Total_num_of_sentence : ' + str(num_sentence))

    return max([len(i) for i in clean_docs])


def clean_document(doc_sentence_list, dataset):
    stop_words = stopwords.words('english')
    stop_words = set(stop_words)
    stemmer = WordNetLemmatizer()

    word_freq = Counter()

    for doc_sentences in doc_sentence_list:
        for sentence in doc_sentences:
            temp = word_tokenize(clean_str(sentence))
            temp = ' '.join([stemmer.lemmatize(word) for word in temp])

            words = temp.split()
            for word in words:
                word_freq[word] += 1

    highbar = word_freq.most_common(10)[-1][1]
    clean_docs = []
    for doc_sentences in doc_sentence_list:
        clean_doc = []
        count_num = 0
        for sentence in doc_sentences:
            temp = word_tokenize(clean_str(sentence))
            temp = ' '.join([stemmer.lemmatize(word) for word in temp])

            words = temp.split()
            doc_words = []
            for word in words:
                if word.isdigit():
                    doc_words.append('digit')
                    continue
                if dataset == 'mr':
                    if word not in stop_words:
                        doc_words.append(word)
                elif (word not in stop_words) and (word_freq[word] >= 5) and (word_freq[word] < highbar):
                    doc_words.append(word)

            clean_doc.append(doc_words)
            count_num += len(doc_words)

            # if dataset == '20ng' and count_num > 2000:
            #     break
        # clean_doc = make_window(clean_doc, 7, inc_sent=True)
        clean_doc = combine_sent(clean_doc)
        clean_docs.append(clean_doc)

    return clean_docs


def combine_sent(sents):
    new_sents = []
    for i in range(len(sents) - 1):
        new_sents.append(sents[i] + sents[i + 1])
    if len(sents) == 1:
        return sents
    return new_sents


def make_window(sents, window_size, inc_sent=False):
    windows = []
    for sent in sents:
        if len(sent) == 0:
            continue
        if len(sent) <= window_size:
            windows.append(sent)
            continue
        if inc_sent:
            windows.append(sent)
        for i in range(0, len(sent) - window_size):
            windows.append(sent[i:i + window_size])

    if len(windows) == 0:
        print('zero size document')
    return windows


def create_word_vectors(doc_content_list, global_word_to_id):
    dim = 300
    global_vocab = [None] * len(global_word_to_id)
    for word in global_word_to_id:
        global_vocab[global_word_to_id[word]] = word

    windows_g = []
    for doc in doc_content_list:
        windows_g.extend(make_window(doc, 7))
        # for window in doc:
        #     windows_g.append(window)

    global_vocab_size = len(global_vocab)

    word_window_freq = {}
    for window in windows_g:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_count = {}
    for window in windows_g:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = global_word_to_id[word_i]
                word_j = window[j]
                word_j_id = global_word_to_id[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    row = []
    col = []
    weight = []

    # pmi as weights
    num_window = len(windows_g)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[global_vocab[i]]
        word_freq_j = word_window_freq[global_vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))

        if pmi <= 0:
            continue
        row.append(i)
        col.append(j)
        weight.append(pmi)

    adj_g = sp.csr_matrix(
        (weight, (row, col)), shape=(global_vocab_size, global_vocab_size))
    svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=42)
    # word_vectors = adj_g + sp.identity(adj_g.shape[0])
    word_vectors = svd.fit_transform(adj_g)
    word_vectors[0] = np.zeros(dim, dtype='float32')

    return word_vectors
