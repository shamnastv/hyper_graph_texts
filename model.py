import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from attention import Attention
from layer import HGNNLayer


def get_features(data, device):
    edge_nums = [d.edge_num for d in data]
    node_nums = [d.node_num for d in data]
    max_e = max(edge_nums)
    max_v = max(node_nums)

    incident_mat = []
    degrees_v = []
    degrees_e = []
    e_masks = []
    v_masks = []
    targets = []
    x = []

    for d in data:
        h = sp.coo_matrix((d.vals, (d.rows, d.cols)), shape=(max_v, max_e))
        incident_mat.append(torch.from_numpy(h.todense()).float())

        v_mask_size = max_v - d.node_num
        e_mask_size = max_e - d.edge_num

        degree_v = d.degrees_v + [0] * v_mask_size
        degree_e = d.degrees_e + [0] * e_mask_size

        degrees_v.append(torch.diag(torch.tensor(degree_v).float()))
        degrees_e.append(torch.diag(torch.tensor(degree_e).float()))

        v_masks.append(torch.tensor([1] * d.node_num + [0] * v_mask_size).float())
        e_masks.append(torch.tensor([1] * d.edge_num + [0] * e_mask_size).float())

        targets.append(d.label)

        x.append(torch.tensor(d.node_ids + [0] * v_mask_size).long())

    incident_mat = torch.stack(incident_mat).to(device)
    degrees_v = torch.stack(degrees_v).to(device)
    degrees_e = torch.stack(degrees_e).to(device)
    x = torch.stack(x).to(device)
    e_masks = torch.stack(e_masks).to(device)
    v_masks = torch.stack(v_masks).to(device)
    targets = torch.tensor(targets).long().to(device)

    return incident_mat, degrees_v, degrees_e, x, e_masks, v_masks, targets


class HGNNModel(nn.Module):
    def __init__(self, args, input_dim, num_classes, word_vectors, device):
        super(HGNNModel, self).__init__()

        self.device = device

        self.word_embeddings = nn.Embedding(word_vectors.shape[0], word_vectors.shape[1], padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_vectors).float())
        self.word_embeddings.weight.requires_grad = True

        self.h_gnn_layers = nn.ModuleList()
        self.linears_prediction = torch.nn.ModuleList()
        self.attention = nn.ModuleList()

        self.num_layers = args.num_layers

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.h_gnn_layers.append(HGNNLayer(args, input_dim, args.hidden_dim))
                # self.linears_prediction.append(nn.Linear(2 * input_dim, num_classes))
                self.linears_prediction.append(nn.Linear(input_dim, num_classes))
                self.attention.append(Attention(input_dim))
            else:
                self.h_gnn_layers.append(HGNNLayer(args, args.hidden_dim, args.hidden_dim))
                # self.linears_prediction.append(nn.Linear(2 * args.hidden_dim, num_classes))
                self.linears_prediction.append(nn.Linear(args.hidden_dim, num_classes))
                self.attention.append(Attention(args.hidden_dim))

        # self.linears_prediction.append(nn.Linear(2 * args.hidden_dim, num_classes))
        self.linears_prediction.append(nn.Linear(args.hidden_dim, num_classes))
        self.attention.append(Attention(args.hidden_dim))

    def forward(self, data):

        incident_mat, degree_v, degree_e, x, e_masks, v_masks, targets = get_features(data, self.device)

        h = self.word_embeddings(x)
        h_cat = [h]

        for layer in range(self.num_layers - 1):
            h = self.h_gnn_layers[layer](incident_mat, degree_v, degree_e, h, e_masks)
            h_cat.append(h)

        pred = 0
        for layer, h in enumerate(h_cat):
            # if layer == 0:
            #     continue
            attn = self.attention[layer](h)
            attn = F.softmax(attn.masked_fill(v_masks.eq(0).unsqueeze(2), -1e9), dim=1)
            doc_embed1 = torch.bmm(attn.transpose(1, 2), h).squeeze(1)

            # masks = v_masks.eq(0).unsqueeze(2).repeat(1, 1, h.shape[2])
            # doc_embed2 = torch.max(h.masked_fill(masks, -1e9), dim=1)[0]

            # pred += self.linears_prediction[layer](torch.cat((doc_embed1, doc_embed2), dim=1))
            pred += self.linears_prediction[layer](doc_embed1)

        # attn = self.attention[self.num_layers - 1](h)
        # attn = F.softmax(attn.masked_fill(v_masks.eq(0).unsqueeze(2), -1e9), dim=1)
        # doc_embed1 = torch.bmm(attn.transpose(1, 2), h).squeeze(1)
        # pred = self.linears_prediction[self.num_layers - 1](doc_embed1)

        return pred, targets
