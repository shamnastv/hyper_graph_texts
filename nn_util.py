import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture


def get_init_embd(data, word_vectors):
    word_embeddings = nn.Embedding(word_vectors.shape[0], word_vectors.shape[1], padding_idx=0)
    word_embeddings.weight.data.copy_(torch.from_numpy(word_vectors).float())
    word_embeddings.weight.requires_grad = False

    init_embed = []
    for d in data:
        word_ids = d.node_ids
        word_weights = [1/len(word_ids)] * len(word_ids)
        word_weights = torch.tensor(word_weights).float().unsqueeze(0)
        word_ids = torch.tensor(word_ids).long()
        word_embds = word_embeddings(word_ids)
        embd = torch.matmul(word_weights, word_embds)
        init_embed.append(embd)

    init_embed = torch.cat(init_embed, dim=0).numpy()
    return init_embed


def get_init_embd2(data, word_vectors):

    num_words = len(word_vectors)
    num_docs = len(data)
    row = []
    col = []
    weight = []

    for i, d in enumerate(data):
        for j, k in enumerate(d.node_ids):
            w = d.tf[j] * d.idf[j]
            row.append(i)
            col.append(k)
            weight.append(w)

    embd = sp.csr_matrix((weight, (row, col)), shape=(num_docs, num_words))
    svd = TruncatedSVD(n_components=200, n_iter=7, random_state=42)
    embd = svd.fit_transform(embd)
    return embd


def clustering(data, num_clusters):
    cluster = KMeans(num_clusters, random_state=0)
    c = cluster.fit(data)
    labels = c.labels_
    # cluster = GaussianMixture(n_components=num_clusters, random_state=0)
    # labels = cluster.fit_predict(data)
    return labels


def split_data(data, clusters):

    new_data = [[] for i in range(max(clusters) + 1)]
    for i, c in enumerate(clusters):
        new_data[c].append(data[i])

    return new_data
