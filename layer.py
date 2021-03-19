import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_sparse import spmm

from mlp import MLP


class GRUCellMod(nn.Module):
    """
    ref https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
    """

    def __init__(self, input_dim, hidden_dim):
        super(GRUCellMod, self).__init__()
        self.W_ir = nn.Linear(input_dim, hidden_dim)
        self.W_hr = nn.Linear(hidden_dim, hidden_dim)
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_hn = nn.Linear(hidden_dim, hidden_dim)
        self.W_iz = nn.Linear(input_dim, hidden_dim)
        self.W_hz = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inp, ht_1):
        # r_t = torch.sigmoid(self.W_ir(inp) + self.W_hr(ht_1))
        z_t = torch.sigmoid(self.W_iz(inp) + self.W_hz(ht_1))
        # n_t = F.leaky_relu(self.W_in(inp) + r_t * self.W_hn(ht_1))
        # h_t = z_t * n_t + (1 - z_t) * ht_1
        h_t = z_t * inp + (1 - z_t) * ht_1
        return h_t


class HGNNLayer(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(HGNNLayer, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.activation = F.leaky_relu
        self.mlp = MLP(args.num_mlp_layers, input_dim, args.hidden_dim, output_dim, args.dropout)
        self.mlp2 = MLP(args.num_mlp_layers, args.hidden_dim, args.hidden_dim, output_dim, args.dropout)
        self.theta_att = nn.Parameter(torch.zeros(input_dim, 1), requires_grad=True)
        self.eps = nn.Parameter(torch.rand(1), requires_grad=True)
        self.batch_norms = nn.BatchNorm1d(output_dim)
        self.batch_norms2 = nn.BatchNorm1d(output_dim)
        self.gru = GRUCellMod(output_dim, output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta_att.size(1))
        self.theta_att.data.uniform_(-stdv, stdv)

    def forward(self, incident_mat_full, degree_v_full, degree_e_full, h, layer):
        # if layer == 1:
        #     h = self.message_passing_1(incident_mat, x, degree_v, degree_e, e_masks)
        # # for i in range(layer + 1):
        # if layer == 0:
        #     h = self.message_passing_2(incident_mat, x, degree_v, degree_e)

        h = self.mlp(h)
        h_n = self.message_passing_3_1(incident_mat_full, h, degree_v_full)
        h_n = self.activation(h_n)
        h_n = self.dropout(h_n)
        h_n = self.batch_norms(h_n)

        h_n = self.mlp2(h_n)
        h_n = self.message_passing_3_2(incident_mat_full, h_n, degree_e_full)
        h_n = self.activation(h_n)
        h_n = self.dropout(h_n)
        h_n = self.batch_norms2(h_n)
        h_n = h_n + self.eps * h

        # h = self.mlp(h)
        # h = self.message_passing_2(incident_mat_full, h, degree_v_full, degree_e_full)
        # h = self.activation(h)
        # h = self.dropout(h)
        # h_n = self.batch_norms(h)

        return h_n

    def message_passing_1(self, incident_mat_full, x, degree_v_full, degree_e_full, e_masks):
        ht_x = spmm(torch.flip(incident_mat_full[0], [0]), incident_mat_full[1],
                    incident_mat_full[2][1], incident_mat_full[2][0], x)
        ht_x = spmm(degree_e_full[0], degree_e_full[1], degree_e_full[2][0], degree_e_full[2][1], ht_x)
        x_theta = torch.matmul(x, self.theta_att)
        ht_x_theta = spmm(torch.flip(incident_mat_full[0], [0]), incident_mat_full[1],
                          incident_mat_full[2][1], incident_mat_full[2][0], x_theta).squeeze(2)
        ht_x_theta = ht_x_theta.masked_fill(e_masks.eq(0), -np.inf)
        hyper_edge_attn = F.softmax(ht_x_theta, dim=1)
        hyper_edge_attn = torch.diag_embed(hyper_edge_attn)
        h1 = torch.bmm(hyper_edge_attn, ht_x)
        h1 = spmm(incident_mat_full[0], incident_mat_full[1], incident_mat_full[2][0], incident_mat_full[2][1], h1)
        h1 = spmm(degree_v_full[0], degree_v_full[1], degree_v_full[2][0], degree_v_full[2][1], h1)
        h1 = spmm(torch.flip(incident_mat_full[0], [0]), incident_mat_full[1],
                  incident_mat_full[2][1], incident_mat_full[2][0], h1)
        h1 = spmm(degree_e_full[0], degree_e_full[1], degree_e_full[2][0], degree_e_full[2][1], h1)
        h2 = self.eps * ht_x
        h = h1 + h2
        h = spmm(incident_mat_full[0], incident_mat_full[1], incident_mat_full[2][0], incident_mat_full[2][1], h)
        h = spmm(degree_v_full[0], degree_v_full[1], degree_v_full[2][0], degree_v_full[2][1], h)
        return h

    def message_passing_2(self, incident_mat_full, x, degree_v_full, degree_e_full):
        h = spmm(degree_v_full[0], degree_v_full[1] ** .5, degree_v_full[2][0], degree_v_full[2][1], x)
        h = spmm(torch.flip(incident_mat_full[0], [0]), incident_mat_full[1],
                 incident_mat_full[2][1], incident_mat_full[2][0], h)
        h = spmm(degree_e_full[0], degree_e_full[1], degree_e_full[2][0], degree_e_full[2][1], h)
        h = spmm(incident_mat_full[0], incident_mat_full[1], incident_mat_full[2][0], incident_mat_full[2][1], h)
        h = spmm(degree_v_full[0], degree_v_full[1] ** .5, degree_v_full[2][0], degree_v_full[2][1], h)
        # h = self.gru(h, x_w)
        h = h + self.eps * x
        return h

    def message_passing_3_1(self, incident_mat_full, h, degree_v_full):
        h = spmm(degree_v_full[0], degree_v_full[1], degree_v_full[2][0], degree_v_full[2][1], h)
        h = spmm(torch.flip(incident_mat_full[0], [0]), incident_mat_full[1],
                 incident_mat_full[2][1], incident_mat_full[2][0], h)
        return h

    def message_passing_3_2(self, incident_mat_full, h, degree_e_full):
        h = spmm(degree_e_full[0], degree_e_full[1], degree_e_full[2][0], degree_e_full[2][1], h)
        h = spmm(incident_mat_full[0], incident_mat_full[1], incident_mat_full[2][0], incident_mat_full[2][1], h)
        return h
