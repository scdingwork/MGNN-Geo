from dgl.nn.pytorch import GraphConv,NNConv
import torch
import torch.nn as nn
#D:\OneDrive - yourdrive\3_人生笔记\4 专业任务（长期笔记分类）\7_论文\1_blackhole_geo\6_延迟估计算法.md
#延迟处理思路
# black_geo_0121.py代码还要小改一下

class MPNN(nn.Module):
    def __init__(self, aggregator_type,node_in_feats, node_hidden_dim, edge_input_dim, edge_hidden_dim,num_step_message_passing,gconv_dp,edge_dp,nn_dp1):
        #原GCN是对基础的GraphConv(in_feats, hidden_size)的包装，多了一个linear_size参数，代表最后卷积层输出后，额外加一层的
        # FC，好得到最终的表示，非常简单。。。
        super(MPNN, self).__init__()
        self.lin0 = nn.Linear(node_in_feats, node_hidden_dim)#65,32
        self.num_step_message_passing=num_step_message_passing#层数 开始测试1层即可
        edge_network = nn.Sequential(
            # edge的处理网络 一般是MLP，输入自然是edge_input_dim，
            # 最后的输出必须是图卷积层的节点输入特征x节点隐藏特征
            # 原文注释 注意in_feats和out_feats就是NNConv(in_feats=node_in_feats,out_feats=node_hidden_dim,...)
            # edge_func : callable activation function/layer
            # Maps each edge feature to a vector of shape
            # ``(in_feats * out_feats)`` as weight to compute messages.
            #我们这里在原始的节点输入dim基础上，通过一层线性层将其转换为指定的维度作为图卷积层的输入node_hidden_dim
            #参考qm9_nn 多层，edgefunc也是一样的 我的想法还没搞清楚；minist则是不一样的；mpnndgl也是一样的

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