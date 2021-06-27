import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_sparse import spmm

from attention import Attention
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
        z_t = torch.sigmoid(self.W_iz(inp) + self.W_hz(ht_1))
        r_t = torch.sigmoid(self.W_ir(inp) + self.W_hr(ht_1))
        n_t = torch.tanh(self.W_in(inp) + self.W_hn(r_t * ht_1))
        h_t = (1 - z_t) * n_t + z_t * ht_1
        return h_t


class HGNNLayer(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(HGNNLayer, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.activation = torch.tanh
        self.mlp1 = MLP(args.num_mlp_layers, input_dim, args.hidden_dim, output_dim, args.dropout)
        self.mlp2 = MLP(args.num_mlp_layers, output_dim, args.hidden_dim, output_dim, args.dropout)
        self.theta_att = nn.Parameter(torch.zeros(output_dim, 1), requires_grad=True)
        # self.theta_att_mlp = Attention(output_dim, activation=torch.tanh)
        self.eps = nn.Parameter(torch.rand(1), requires_grad=True)
        self.batch_norms = nn.BatchNorm1d(output_dim)
        self.batch_norms2 = nn.BatchNorm1d(output_dim)
        self.gru = GRUCellMod(input_dim, output_dim)
        self.att_1 = Attention(output_dim * 2, activation=F.leaky_relu, num_layers=2)
        self.att_2 = Attention(output_dim * 2, activation=F.leaky_relu, num_layers=2)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta_att.size(1))
        self.theta_att.data.uniform_(-stdv, stdv)

    def forward(self, incident_mat_full, degree_v_full, degree_e_full, h, layer, sent_mat_full):
        # h = self.mlp1(h)
        # h_n = self.message_passing_1(incident_mat_full, h, degree_v_full, degree_e_full)
        # h_n = self.activation(h_n)
        # h_n = self.dropout(h_n)
        # h_n = self.batch_norms(h_n)

        # h = self.mlp1(h)
        # h = self.message_passing_2(incident_mat_full, h, degree_v_full, degree_e_full)
        # h = self.activation(h)
        # h_n = self.dropout(h)
        # h_n = self.batch_norms(h_n)

        h_m = self.mlp1(h)
        h_n = self.message_passing_3_1(incident_mat_full, h_m, degree_e_full)
        h_n = spmm(sent_mat_full[0], sent_mat_full[1], sent_mat_full[2][0], sent_mat_full[2][1], h_n)
        h_n = self.batch_norms2(h_n)
        h_n = self.activation(h_n)
        # h_n = F.leaky_relu(h_n, negative_slope=0.2)
        h_n = self.dropout(h_n)

        h_n = self.mlp2(h_n)
        h_n = self.message_passing_3_2(incident_mat_full, h_n, degree_v_full)
        h_n = self.batch_norms(h_n)
        h_n = self.activation(h_n)
        # h_n = F.leaky_relu(h_n, negative_slope=0.2)
        h_n = self.dropout(h_n)
        # h_n = h_n + self.eps * h_m
        h_n = self.gru(h, h_n)
        # h_n = self.dropout(h_n)

        return h_n

    def message_passing_1(self, incident_mat_full, x, degree_v_full, degree_e_full):
        ht_x = spmm(torch.flip(incident_mat_full[0], [0]), incident_mat_full[1],
                    incident_mat_full[2][1], incident_mat_full[2][0], x)
        ht_x = spmm(degree_e_full[0], degree_e_full[1], degree_e_full[2][0], degree_e_full[2][1], ht_x)
        x_theta = torch.matmul(x, self.theta_att)
        # x_theta = self.theta_att_mlp(x)
        ht_x_theta = spmm(torch.flip(incident_mat_full[0], [0]), incident_mat_full[1],
                          incident_mat_full[2][1], incident_mat_full[2][0], x_theta).squeeze(1)
        hyper_edge_attn = torch.sigmoid(ht_x_theta)
        hyper_edge_attn = torch.diag_embed(hyper_edge_attn)
        h1 = torch.mm(hyper_edge_attn, ht_x)
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
        # h = self.gru(h, x)
        h = h + self.eps * x
        return h

    def message_passing_3_1(self, incident_mat_full, h, degree_e_full):
        idx = torch.flip(incident_mat_full[0], [0])
        h_t = spmm(idx, incident_mat_full[1], incident_mat_full[2][1], incident_mat_full[2][0], h)
        h_t = spmm(degree_e_full[0], degree_e_full[1], degree_e_full[2][0], degree_e_full[2][1], h_t)

        attn_inpt = torch.cat((h[idx[1]], h_t[idx[0]]), dim=1)
        # attn_inpt = F.leaky_relu(attn_inpt, negative_slope=0.2)
        att = self.att_1(attn_inpt).squeeze(1)

        with torch.no_grad():
            maximum = torch.max(att)
        att = att - maximum

        att = torch.exp(att)
        assert not torch.isnan(att).any()

        ones = torch.ones(size=(h.shape[0], 1), device=h.device)
        pooled = spmm(idx, att, incident_mat_full[2][1], incident_mat_full[2][0], h)
        row_sum = spmm(idx, att, incident_mat_full[2][1], incident_mat_full[2][0], ones) + 1e-20
        h_e = pooled.div(row_sum)

        return h_e

    def message_passing_3_2(self, incident_mat_full, h_e, degree_v_full):
        idx = incident_mat_full[0]
        h_t = spmm(idx, incident_mat_full[1], incident_mat_full[2][0], incident_mat_full[2][1], h_e)
        h_t = spmm(degree_v_full[0], degree_v_full[1], degree_v_full[2][0], degree_v_full[2][1], h_t)

        attn_inpt = torch.cat((h_e[idx[1]], h_t[idx[0]]), dim=1)
        # attn_inpt = F.leaky_relu(attn_inpt, negative_slope=0.2)
        att = self.att_2(attn_inpt).squeeze(1)

        with torch.no_grad():
            maximum = torch.max(att)
        att = att - maximum

        att = torch.exp(att)
        assert not torch.isnan(att).any()

        ones = torch.ones(size=(h_e.shape[0], 1), device=h_e.device)
        pooled = spmm(idx, att, incident_mat_full[2][0], incident_mat_full[2][1], h_e)
        row_sum = spmm(idx, att, incident_mat_full[2][0], incident_mat_full[2][1], ones) + 1e-20
        h_v = pooled.div(row_sum)

        return h_v
