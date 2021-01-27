import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

    def forward(self, incident_mat, degree_v, degree_e, h, e_masks, layer):

        # if layer == 1:
        #     h = self.message_passing_1(incident_mat, x, degree_v, degree_e, e_masks)
        # # for i in range(layer + 1):
        # if layer == 0:
        #     h = self.message_passing_2(incident_mat, x, degree_v, degree_e)

        h = self.mlp(h)
        h_n = self.message_passing_3_1(incident_mat, h, degree_v)
        h_n = self.activation(h_n)
        h_n = self.dropout(h_n)
        h_n = self.batch_norms(h_n.transpose(1, 2)).transpose(1, 2)

        h_n = self.mlp2(h_n)
        h_n = self.message_passing_3_2(incident_mat, h_n, degree_e)
        h_n = self.activation(h_n)
        h_n = self.dropout(h_n)
        h_n = self.batch_norms2(h_n.transpose(1, 2)).transpose(1, 2)
        h_n = h_n + self.eps * h
        # h = self.message_passing_2(incident_mat, h, degree_v, degree_e)
        # h = self.activation(h)
        # h = self.dropout(h)
        # h = self.batch_norms(h.transpose(1, 2)).transpose(1, 2)

        return h_n

    def message_passing_1(self, incident_mat, x, degree_v, degree_e, e_masks):
        ht_x = torch.bmm(incident_mat.transpose(1, 2), x)
        ht_x = torch.bmm(degree_e, ht_x)
        x_theta = torch.matmul(x, self.theta_att)
        ht_x_theta = torch.bmm(incident_mat.transpose(1, 2), x_theta).squeeze(2)
        ht_x_theta = ht_x_theta.masked_fill(e_masks.eq(0), -np.inf)
        hyper_edge_attn = F.softmax(ht_x_theta, dim=1)
        hyper_edge_attn = torch.diag_embed(hyper_edge_attn)
        h1 = torch.bmm(hyper_edge_attn, ht_x)
        h1 = torch.bmm(incident_mat, h1)
        h1 = torch.bmm(degree_v, h1)
        h1 = torch.bmm(incident_mat.transpose(1, 2), h1)
        h1 = torch.bmm(degree_e, h1)
        h2 = self.eps * ht_x
        h = h1 + h2
        h = torch.bmm(incident_mat, h)
        h = torch.bmm(degree_v, h)
        return h

    def message_passing_2(self, incident_mat, x, degree_v, degree_e):
        degree_v_root = degree_v ** .5
        h = torch.bmm(degree_v_root, x)
        h = torch.bmm(incident_mat.transpose(1, 2), h)
        h = torch.bmm(degree_e, h)
        h = torch.bmm(incident_mat, h)
        h = torch.bmm(degree_v_root, h)
        # h = self.gru(h, x_w)
        h = h + self.eps * x
        return h

    def message_passing_3_1(self, incident_mat, h, degree_v):
        h = torch.bmm(degree_v, h)
        h = torch.bmm(incident_mat.transpose(1, 2), h)
        return h

    def message_passing_3_2(self, incident_mat, h, degree_e):
        h = torch.bmm(degree_e, h)
        h = torch.bmm(incident_mat, h)
        return h
