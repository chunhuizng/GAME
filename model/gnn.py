from dgl.nn.pytorch import GATConv
from grb.model.dgl import GAT
import torch.nn.functional as F
from model.moe_tools import GATConv_moe, GCNConv_moe
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


from grb.model.torch import GCN
from grb.model.torch.gcn import GCNConv
from grb.utils.normalize import GCNAdjNorm
import torch.nn as nn
import torch.nn.functional as F

class GAT_moe(GAT):
    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 n_layers,
                 n_heads,
                 activation=F.leaky_relu,
                 layer_norm=False,
                 feat_norm=None,
                 adj_norm_func=None,
                 feat_dropout=0.0,
                 attn_dropout=0.0,
                 residual=False,
                 dropout=0.0, num_experts=4, noisy_gating=True, k=1):
        super(GAT_moe, self).__init__(in_features=in_features,
                 out_features=out_features,
                 hidden_features=hidden_features,
                 n_layers=n_layers,
                 n_heads=n_heads,
                 activation=activation,
                 layer_norm=layer_norm,
                 feat_norm=feat_norm,
                 adj_norm_func=adj_norm_func,
                 feat_dropout=feat_dropout,
                 attn_dropout=attn_dropout,
                 residual=residual,
                 dropout=dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_features) is int:
            hidden_features = [hidden_features] * (n_layers - 1)
        elif type(hidden_features) is list or type(hidden_features) is tuple:
            assert len(hidden_features) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."
        n_features = [in_features] + hidden_features + [out_features]

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if layer_norm:
                if i == 0:
                    self.layers.append(nn.LayerNorm(n_features[i]))
                else:
                    self.layers.append(nn.LayerNorm(n_features[i] * n_heads))
            self.layers.append(GATConv_moe(in_feats=n_features[i] * n_heads if i != 0 else n_features[i],
                                       out_feats=n_features[i + 1],
                                       num_heads=n_heads if i != n_layers - 1 else 1,
                                       feat_drop=feat_dropout if i != n_layers - 1 else 0.0,
                                       attn_drop=attn_dropout if i != n_layers - 1 else 0.0,
                                       residual=residual if i != n_layers - 1 else False,
                                       activation=activation if i != n_layers - 1 else None,
                                       num_experts=num_experts, noisy_gating=noisy_gating, k=k))
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

class gcn_bn(GCN):
    def __init__(self,in_features,
                 out_features,
                 hidden_features,
                 n_layers,
                 activation=F.relu,
                 layer_norm=False,
                 residual=False,
                 feat_norm=None,
                 adj_norm_func=GCNAdjNorm,
                 dropout=0.0):
        super(gcn_bn, self).__init__(in_features=in_features, out_features=out_features, hidden_features=hidden_features, n_layers=n_layers, activation=activation, layer_norm=layer_norm, residual=residual, feat_norm=feat_norm, adj_norm_func=adj_norm_func, dropout=dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_features) is int:
            hidden_features = [hidden_features] * (n_layers - 1)
        elif type(hidden_features) is list or type(hidden_features) is tuple:
            assert len(hidden_features) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."
        n_features = [in_features] + hidden_features + [out_features]

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if layer_norm:
                self.layers.append(nn.BatchNorm1d(n_features[i]))
            self.layers.append(GCNConv(in_features=n_features[i],
                                       out_features=n_features[i + 1],
                                       activation=activation if i != n_layers - 1 else None,
                                       residual=residual if i != n_layers - 1 else False,
                                       dropout=dropout if i != n_layers - 1 else 0.0))
        self.reset_parameters()

    def forward(self, x, adj):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
            else:
                x = layer(x, adj)
        return x


class GCN_bn_moe(gcn_bn):
    def __init__(self, in_features, out_features, hidden_features, n_layers,
                 activation=F.relu,
                 layer_norm=False,
                 residual=False,
                 feat_norm=None,
                 adj_norm_func=GCNAdjNorm,
                 dropout=0.0,
                 num_experts=4, noisy_gating=True, k=1):
        super(gcn_bn, self).__init__(in_features=in_features, out_features=out_features, hidden_features=hidden_features, n_layers=n_layers, activation=activation, layer_norm=layer_norm, residual=residual, feat_norm=feat_norm, adj_norm_func=adj_norm_func, dropout=dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_features) is int:
            hidden_features = [hidden_features] * (n_layers - 1)
        elif type(hidden_features) is list or type(hidden_features) is tuple:
            assert len(hidden_features) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."
        n_features = [in_features] + hidden_features + [out_features]

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if layer_norm:
                self.layers.append(nn.BatchNorm1d(n_features[i]))
            self.layers.append(GCNConv_moe(in_features=n_features[i],
                                       out_features=n_features[i + 1],
                                       activation=activation if i != n_layers - 1 else None,
                                       residual=residual if i != n_layers - 1 else False,
                                       dropout=dropout if i != n_layers - 1 else 0.0,
                                       num_experts=num_experts, noisy_gating=noisy_gating, k=k))
        # self.reset_parameters()

class GCN_ln_moe(gcn_bn):
    def __init__(self, in_features, out_features, hidden_features, n_layers,
                 activation=F.relu,
                 layer_norm=False,
                 residual=False,
                 feat_norm=None,
                 adj_norm_func=GCNAdjNorm,
                 dropout=0.0,
                 num_experts=4, noisy_gating=True, k=1):
        super(gcn_bn, self).__init__(in_features=in_features, out_features=out_features, hidden_features=hidden_features, n_layers=n_layers, activation=activation, layer_norm=layer_norm, residual=residual, feat_norm=feat_norm, adj_norm_func=adj_norm_func, dropout=dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_features) is int:
            hidden_features = [hidden_features] * (n_layers - 1)
        elif type(hidden_features) is list or type(hidden_features) is tuple:
            assert len(hidden_features) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."
        n_features = [in_features] + hidden_features + [out_features]

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if layer_norm:
                self.layers.append(nn.LayerNorm(n_features[i]))
            self.layers.append(GCNConv_moe(in_features=n_features[i],
                                       out_features=n_features[i + 1],
                                       activation=activation if i != n_layers - 1 else None,
                                       residual=residual if i != n_layers - 1 else False,
                                       dropout=dropout if i != n_layers - 1 else 0.0,
                                       num_experts=num_experts, noisy_gating=noisy_gating, k=k))

    def forward(self, x, adj):
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(x, adj)
        return x