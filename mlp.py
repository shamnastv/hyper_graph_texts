import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout, bias=True):

        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.linears = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linears.append(nn.Linear(input_dim, output_dim, bias=bias))
        else:
            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=bias))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, h):
        for layer in range(self.num_layers - 1):
            h = self.linears[layer](h)
            h = F.leaky_relu(h)
            h = self.dropout(h)
            # h = self.batch_norms[layer](h)
        return self.linears[self.num_layers - 1](h)
