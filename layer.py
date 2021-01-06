import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mlp import MLP


class HGNNLayer(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(HGNNLayer, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.activation = F.leaky_relu
        self.mlp = MLP(args.num_mlp_layers, input_dim, args.hidden_dim, output_dim, args.dropout)
        self.theta_att = nn.Parameter(torch.zeros(input_dim, 1), requires_grad=True)
        self.eps = nn.Parameter(torch.rand(1), requires_grad=True)
        self.reset_parameters()

        self.batch_norms = nn.BatchNorm1d(output_dim)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta_att.size(1))
        self.theta_att.data.uniform_(-stdv, stdv)

    def forward(self, incident_mat, degree_v, degree_e, x, e_masks):

        x_w = self.mlp(x)
        ht_x_w = torch.bmm(incident_mat.transpose(1, 2), x_w)
        x_theta = torch.matmul(x, self.theta_att)
        ht_x_theta = torch.bmm(incident_mat.transpose(1, 2), x_theta).squeeze(2)
        ht_x_theta = ht_x_theta.masked_fill(e_masks.eq(0), -1e9)
        hyper_edge_attn = F.softmax(ht_x_theta, dim=1)
        hyper_edge_attn = torch.diag_embed(hyper_edge_attn)

        h1 = torch.bmm(hyper_edge_attn, ht_x_w)
        h1 = torch.bmm(incident_mat, h1)
        h1 = torch.bmm(degree_v, h1)
        h1 = torch.bmm(incident_mat.transpose(1, 2), h1)
        h1 = torch.bmm(degree_e, h1)

        h2 = self.eps * ht_x_w

        h = h1 + h2

        h = torch.bmm(incident_mat, h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.batch_norms(h.transpose(1, 2)).transpose(1, 2)

        return h
