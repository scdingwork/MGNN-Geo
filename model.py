from dgl.nn.pytorch import GraphConv,NNConv
import torch
import torch.nn as nn


class MPNN(nn.Module):
    def __init__(self, aggregator_type,node_in_feats, node_hidden_dim, edge_input_dim, edge_hidden_dim,num_step_message_passing,gconv_dp,edge_dp,nn_dp1):
        super(MPNN, self).__init__()
        self.lin0 = nn.Linear(node_in_feats, node_hidden_dim)#65,32
        self.num_step_message_passing=num_step_message_passing#
        edge_network = nn.Sequential(
          
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=edge_dp),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim),
            nn.Dropout(p=edge_dp))#1-4-32x32

        self.conv = NNConv(in_feats=node_hidden_dim,#32
               out_feats=node_hidden_dim,#32
               edge_func=edge_network,#32x32
               aggregator_type=aggregator_type)

        self.y_linear = nn.Linear(node_hidden_dim, 2)#4-4
        self.y_linear2 = nn.Linear(node_hidden_dim, 1)  # 4-4
        self.bn = nn.BatchNorm1d(node_hidden_dim)

        self.gnn_dropout = nn.Dropout(p=gconv_dp)#dropout
        self.nn_dropout = nn.Dropout(p=nn_dp1)
        # self.nn_dropout2 = nn.Dropout(p=nn_dp2)

    def forward(self, g, n_feat, e_feat):
        out = torch.relu(self.lin0(n_feat))  # (B1, H1)

        for i in range(self.num_step_message_passing):
            out = torch.relu(self.conv(g, out, e_feat))
            # out = self.gnn_dropout(out)# (B1, H1)
            # temp_out = torch.dstack((out, out))

        # temp = torch.mean(temp_out, dim=2)
        y_bn = self.bn(out)
        y_sigmoid = torch.sigmoid(self.y_linear(y_bn))
        y_sigmoid2 = torch.sigmoid(self.y_linear2(y_bn))

        return y_sigmoid,y_sigmoid2
