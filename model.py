import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from torch_sparse import spmm

from attention import Attention
from layer import HGNNLayer


def get_features(data, device):
    targets = []
    x = []
    tf = []
    idf = []

    sent_mat = []
    sent_const = 2
    incident_mat = []
    v_start = 0
    e_start = 0
    graph_pool = []
    max_pool_idx = []
    max_nodes = max([d.node_num for d in data])
    # word_dict = {}
    clusters = [d.cluster for d in data]
    # clusters = [0] * len(data)
    word_dict_ls = [{} for _ in range(max(clusters) + 1)]

    for i, d in enumerate(data):
        c = clusters[i]
        for j in range(len(d.vals)):
            incident_mat.append([d.rows[j] + v_start, d.cols[j] + e_start, d.vals[j]])
        for j, w in enumerate(d.node_ids):
            if w != 0 and c >= 0:
                if w in word_dict_ls[c]:
                    word_dict_ls[c][w].append(v_start + j)
                else:
                    word_dict_ls[c][w] = [v_start + j]
        for m in range(d.edge_num):
            if m - sent_const + 1 >= 0:
                left = (sent_const - 1) * sent_const / 2
            else:
                left = (sent_const - 1) * sent_const / 2 - (sent_const - m) * (sent_const - m - 1)/2
            if m + sent_const - 1 <= d.edge_num:
                right = (sent_const - 1) * sent_const / 2
            else:
                right = (sent_const-1) * sent_const / 2 - (m + sent_const - d.edge_num) * (m + sent_const - 1 - d.edge_num - 1) / 2

            s = left + right + sent_const * 20

            for j in range(sent_const):
                if j == 0:
                    sent_mat.append([e_start + m, e_start + m - j, sent_const * 20 / s])
                else:
                    if m - j >= 0:
                        sent_mat.append([e_start + m, e_start + m - j, (sent_const - j) / s])
                    if m + j < d.edge_num:
                        sent_mat.append([e_start + m, e_start + m + j, (sent_const - j) / s])

        graph_pool.extend([[i, j, 1/d.node_num] for j in range(v_start, v_start + d.node_num, 1)])
        max_pool_idx.append([j for j in range(v_start, v_start + d.node_num, 1)] + [-1] * (max_nodes - d.node_num))
        v_start += d.node_num
        e_start += d.edge_num
        x.extend(d.node_ids)
        targets.append(d.label)
        tf.extend(d.tf)
        idf.extend(d.idf)

    # inter graph edges
    for word_dict in word_dict_ls:
        for w in word_dict:
            if len(word_dict[w]) > 1:
                for i in word_dict[w]:
                    incident_mat.append([i, e_start, 1])
                sent_mat .append([e_start, e_start, 1])
                e_start += 1

    num_v = v_start
    num_e = e_start

    degrees_v = [0] * num_v
    degrees_e = [0] * num_e

    for e in incident_mat:
        degrees_v[e[0]] += 1
        degrees_e[e[1]] += 1

    degrees_e = [1 / i if i != 0 else 0 for i in degrees_e]
    degrees_v = [1 / i if i != 0 else 0 for i in degrees_v]

    degrees_e = torch.tensor(degrees_e).float().to(device)
    degrees_e_shape = len(degrees_e)
    degrees_e_idx_0 = torch.arange(degrees_e_shape).long()
    degrees_e_idx_1 = torch.arange(degrees_e_shape).long()
    degrees_e_idx = torch.stack((degrees_e_idx_0, degrees_e_idx_1), dim=0).to(device)
    degrees_e_full = (degrees_e_idx, degrees_e, torch.Size([degrees_e_shape, degrees_e_shape]))

    degrees_v = torch.tensor(degrees_v).float().to(device)
    degrees_v_shape = len(degrees_v)
    degrees_v_idx_0 = torch.arange(degrees_v_shape).long()
    degrees_v_idx_1 = torch.arange(degrees_v_shape).long()
    degrees_v_idx = torch.stack((degrees_v_idx_0, degrees_v_idx_1), dim=0).to(device)
    degrees_v_full = (degrees_v_idx, degrees_v, torch.Size([degrees_v_shape, degrees_v_shape]))

    targets = torch.tensor(targets).long().to(device)

    x = torch.tensor(x, device=device).long()

    tf = torch.tensor(tf, device=device).float()
    idf = torch.tensor(idf, device=device).float()
    tf_idf = torch.stack((tf, idf), dim=0).transpose(0, 1)

    graph_pool = torch.tensor(graph_pool).float().transpose(0, 1).to(device)
    graph_pool_shape = torch.Size([len(data), len(x)])
    graph_pool_idx = graph_pool[:2].long()
    graph_pool = graph_pool[2]
    graph_pool_full = (graph_pool_idx, graph_pool, graph_pool_shape)

    max_pool_idx = torch.tensor(max_pool_idx).long().to(device)

    incident_mat = torch.tensor(incident_mat).long().transpose(0, 1).to(device)
    incident_mat_shape = torch.Size([v_start, e_start])
    incident_mat_full = (incident_mat[:2], incident_mat[2].float(), incident_mat_shape)

    sent_mat = torch.tensor(sent_mat).float().transpose(0, 1).to(device)
    sent_mat_shape = torch.Size([e_start, e_start])
    sent_mat_full = (sent_mat[:2].long(), sent_mat[2], sent_mat_shape)

    return incident_mat_full, graph_pool_full, degrees_v_full, degrees_e_full, x, targets, max_pool_idx, tf_idf, sent_mat_full


class HGNNModel(nn.Module):
    def __init__(self, args, input_dim, num_classes, word_vectors, device):
        super(HGNNModel, self).__init__()

        self.device = device

        self.word_embeddings = nn.Embedding(word_vectors.shape[0], word_vectors.shape[1], padding_idx=0)
        self.word_embeddings.weight.requires_grad = True
        if not args.random_vec:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(word_vectors).float())
            self.word_embeddings.weight.requires_grad = False

        self.h_gnn_layers = nn.ModuleList()
        self.linears_prediction = torch.nn.ModuleList()

        self.num_layers = args.num_layers
        self.graph_pool_layer = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.h_gnn_layers.append(HGNNLayer(args, input_dim, args.hidden_dim))
                self.linears_prediction.append(nn.Linear(2 * input_dim, num_classes))
                # self.linears_prediction.append(nn.Linear(input_dim, num_classes))
                self.graph_pool_layer.append(Attention(input_dim + 2))
            else:
                self.h_gnn_layers.append(HGNNLayer(args, args.hidden_dim, args.hidden_dim))
                self.linears_prediction.append(nn.Linear(2 * args.hidden_dim, num_classes))
                # self.linears_prediction.append(nn.Linear(args.hidden_dim, num_classes))
                self.graph_pool_layer.append(Attention(args.hidden_dim + 2))

        self.linears_prediction.append(nn.Linear(2 * args.hidden_dim, num_classes))
        # self.linears_prediction.append(nn.Linear(args.hidden_dim, num_classes))
        self.graph_pool_layer.append(Attention(args.hidden_dim + 2))

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, data):

        incident_mat_full, graph_pool_full, degrees_v_full, degrees_e_full, x, targets, max_pool_idx, tf_idf, sent_mat_full\
            = get_features(data, self.device)

        h = self.word_embeddings(x)
        h = self.dropout(h)
        # h_cat = [self.dropout(h)]
        h_cat = [h]

        for layer in range(self.num_layers - 1):
            h = self.h_gnn_layers[layer](incident_mat_full, degrees_v_full, degrees_e_full, h, layer, sent_mat_full)
            h_cat.append(h)

        pred = 0
        pooled_h_ls = []
        for layer, h in enumerate(h_cat):
            # if layer == 0:
            #     continue
            h_t = torch.cat((h, tf_idf), dim=1)
            elem_gp = self.graph_pool_layer[layer](h_t).squeeze(1)

            with torch.no_grad():
                maximum = torch.max(elem_gp)
            elem_gp = elem_gp - maximum
            elem_gp = torch.exp(elem_gp)
            assert not torch.isnan(elem_gp).any()
            # elem_gp = elem_gp * tf_idf[:, 0] * tf_idf[:, 1]

            row_sum = spmm(graph_pool_full[0], elem_gp, graph_pool_full[2][0], graph_pool_full[2][1],
                           torch.ones(size=(h.shape[0], 1), device=self.device))

            pooled_h = spmm(graph_pool_full[0], elem_gp, graph_pool_full[2][0], graph_pool_full[2][1], h)
            # assert not torch.isnan(pooled_h).any()

            pooled_h = pooled_h.div(row_sum + 1e-10)
            assert not torch.isnan(pooled_h).any()

            h = torch.cat([h, torch.ones(1, h.shape[1], device=self.device) * -1e9], dim=0)
            max_pooled = torch.max(h[max_pool_idx], keepdim=False, dim=1)[0]
            # pooled_h = pooled_h + max_pooled
            pooled_h = torch.cat((pooled_h, max_pooled), dim=1)

            pred += self.linears_prediction[layer](pooled_h)
            pooled_h_ls.append(pooled_h)

        pooled_h_ls = torch.cat(pooled_h_ls, dim=1)
        return pred, targets, pooled_h_ls
