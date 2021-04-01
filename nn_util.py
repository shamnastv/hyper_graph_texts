import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from sklearn.cluster import KMeans


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

    init_embed = torch.cat(init_embed, dim=0)
    return init_embed


def clustering(data, num_clusters):
    cluster = KMeans(num_clusters)
    c = cluster.fit(data)
    return c.labels_


def split_data(data, clusters):

    new_data = [[] for i in range(max(clusters) + 1)]
    for i, c in enumerate(clusters):
        new_data[c].append(data[i])

    return new_data