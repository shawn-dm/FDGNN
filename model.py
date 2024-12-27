from torch.nn import Linear
import torch.nn.functional as F
from utils import *
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import GINConv, SAGEConv,GCNConv
from torch.nn.utils import spectral_norm
from torch_geometric.nn import APPNP as APPNP_base


class channel_masker(nn.Module):
    def __init__(self, args):
        super(channel_masker, self).__init__()

        self.weights = nn.Parameter(torch.distributions.Uniform(
            0, 1).sample((args.num_features, 2)))

    def reset_parameters(self):
        self.weights = torch.nn.init.xavier_uniform_(self.weights)

    def forward(self):
        return self.weights


class MLP_discriminator(torch.nn.Module):
    def __init__(self, args):
        super(MLP_discriminator, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, 1)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None, mask_node=None):
        h = self.lin(h)

        return torch.sigmoid(h)


class GCN_encoder_scatter(torch.nn.Module):
    def __init__(self, args):
        super(GCN_encoder_scatter, self).__init__()

        self.args = args

        self.lin = Linear(args.num_features, args.hidden, bias=False)

        self.bias = Parameter(torch.Tensor(args.hidden))

    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):
            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

        # self.lin.weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.lin.weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0.0)

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.lin(x)
        h = propagate2(h, edge_index) + self.bias

        return h

class MLP_encoder(torch.nn.Module):
    def __init__(self, args):
        super(MLP_encoder, self).__init__()
        self.args = args

        self.lin = Linear(args.num_features, args.hidden)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index=None, mask_node=None):
        h = self.lin(x)

        return h

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNEncoder, self).__init__()
        self.conv0 = GCNConv(in_channels, hidden_channels)
        self.transition = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.ReLU()
            # nn.BatchNorm1d(hidden_channels)
        )
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def reset_parameters(self):
        self.conv0.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


    def forward(self, x, edge_index):
        x = self.conv0(x, edge_index)
        x = self.transition(x)
        # x = F.normalize(x, p=2, dim=1) * 1.5
        x_ = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x_, x



class GIN_encoder(nn.Module):
    def __init__(self, args):
        super(GIN_encoder, self).__init__()

        self.args = args

        self.mlp = nn.Sequential(
            nn.Linear(args.num_features, args.hidden),
            # nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            # nn.Linear(args.hidden, args.hidden),
        )

        self.conv = GINConv(self.mlp)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.conv(x, edge_index)
        return h


class SAGEencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SAGEencoder, self).__init__()


        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.ReLU()
            # nn.BatchNorm1d(hidden_channels),

        )
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2.aggr = 'mean'
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3.aggr = 'mean'

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        h_ = self.conv2(x, edge_index)
        h = self.conv3(x, edge_index)
        return h_,h


class APPNPencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(APPNPencoder, self).__init__()

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.conv1 = APPNP_base(K=1, alpha=0.1)
        self.conv2 = APPNP_base(K=1, alpha=0.1)


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        # x = F.relu(self.lin1(x))
        x = self.lin1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index)
        x_ = self.conv2(x, edge_index)

        return x,x_

class MLP_classifier(torch.nn.Module):
    def __init__(self, args):
        super(MLP_classifier, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, args.num_classes)

    def clip_parameters(self):
        for p in self.lin.parameters():
            p.data.clamp_(-self.args.clip_c, self.args.clip_c)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None):
        h = self.lin(h)

        return h
