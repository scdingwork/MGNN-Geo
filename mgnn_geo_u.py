# -*- coding: utf-8 -*-
import os
import numpy as np
import dgl
import random as rd
import torch
import torch.nn as nn
import itertools
from math import radians, cos, sin, asin, sqrt
import argparse
'''
shichang ding
scdingwork@outlook.com
MGNN-Geo for unreachable IP geolocation
'''
parser = argparse.ArgumentParser(description="Run mgnn_geo.")
parser.add_argument('--epochs', type=int, default=1715+1,
                    help='Number of epochs.')
parser.add_argument('--gpuid', type=int, default=0)
parser.add_argument('--node_embedding_size', type=int, default=256)
parser.add_argument('--node_hidden_size', type=int, default=64)
parser.add_argument('--edge_hidden_size', type=int, default=16)
parser.add_argument('--fc_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.0010, help='Learning rate.')
parser.add_argument('--aggregator_type', type=str, default='sum', help='max,sum,mean')  # mean”、“sum”、“max”、“min”
parser.add_argument('--print_flag', type=int, default=1)
parser.add_argument('--num_step_message_passing', type=int, default=3)  # MP depth
parser.add_argument('--regs', type=float, default=0.01)  # l2
parser.add_argument('--gnn_dropout', type=float, default=0.0)  # not used
parser.add_argument('--edge_dropout', type=float, default=0.0)  # not used
parser.add_argument('--nn_dropout1', type=float, default=0.0)  # not used
parser.add_argument('--nn_dropout2', type=float, default=0.0)  # not used
parser.add_argument('--loss1_weight', type=float, default=1)  # importance of auxilliary tasks

args = parser.parse_args()
aggregator_type = args.aggregator_type
node_embedding_size = args.node_embedding_size  # [64,32,16,4,1]
node_hidden_size = args.node_hidden_size
edge_hidden_size = args.edge_hidden_size
fc_size = args.fc_size
regs = args.regs

gnn_dropout = args.gnn_dropout
edge_dropout = args.edge_dropout
nn_dropout1 = args.nn_dropout1
nn_dropout2 = args.nn_dropout2
loss1_weight = args.loss1_weight

gpu_id = args.gpuid
epochs = args.epochs
lr = args.lr
print_flag = args.print_flag
num_step_message_passing = args.num_step_message_passing
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

def setup_seed(seed):
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rd.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

setup_seed(0)

def geodistance(lng1, lat1, lng2, lat2):
    # lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  #6371km
    distance = round(distance / 1000, 3)
    return distance

node_id_delay = {}
node_id_ip ={}
def get_node_attr2(filepath):
    fr1 = open(filepath, 'r')
    node_list = []
    node_attr_list = []
    for line in fr1.readlines():
        str_list = line.strip('\r\n').split(sep=',')
        temp_ip = str(str_list[0])
        temp_nodeID = int(str_list[1])
        node_id_ip[temp_nodeID]=temp_ip
        temp_node_attr = list(map(eval, str_list[2:]))
        temp_node_delay = str_list[2]
        node_id_delay[temp_nodeID] = float(temp_node_delay)
        node_list.append(temp_nodeID)
        node_attr_list.append(temp_node_attr)
    node = np.array(node_list)
    node_attr_array = np.array(node_attr_list)  # .reshape(-1, 1)
    return node_attr_array

def build_graph2(filepath):  # 
    # 
    fr1 = open(filepath, 'r')
    src_list = []
    dst_list = []  # 
    edge_attr_list = []
    for line in fr1.readlines():
        str_list = line.strip('\r\n').split(sep=',')
        temp_src = int(str_list[0])
        temp_dst = int(str_list[1])
        # temp_attr = float(str_list[2])
        temp_node_attr = list(map(eval, str_list[2:]))
        temp_node_delay = str_list[2]

        src_list.append(temp_src)
        dst_list.append(temp_dst)
        edge_attr_list.append(temp_node_attr)

        # 
        src_list.append(temp_dst)
        dst_list.append(temp_src)
        edge_attr_list.append(temp_node_attr)

    src = np.array(src_list)
    dst = np.array(dst_list)
    edge_attr_array = np.array(edge_attr_list)  # .reshape(-1, 1)#

    # 
    # u = np.concatenate([src, dst])#u和v？
    # v = np.concatenate([dst, src])
    u = src  # u和v？
    v = dst
    # Construct a DGLGraph
    return dgl.DGLGraph((u, v)), edge_attr_array  # 


from sklearn import preprocessing

target_scaler1 = preprocessing.MinMaxScaler()
target_scaler2 = preprocessing.MinMaxScaler()
new_label_lat_array = []
new_label_lon_array = []
label_array = []
target_lat_label_dict = {}
target_lon_label_dict = {}
new_target_lat_label_dict = {}
new_target_lon_label_dict = {}

def train_test_val_new_targetscaler2(filepath):#
    fr3 = open(filepath, 'r', encoding='UTF-8')
    target_lat_label_dict = {}
    target_lon_label_dict = {}
    target_delay_label_dict={}

    label_lat_array = []
    label_lon_array = []
    label_delay_array=[]
    for line in fr3.readlines():
        str_list = line.strip('\r\n').split(sep=',')
        node_id = int(str_list[1])
        city_label = str_list[2]
        lat_label = float(str_list[4])
        lon_label = float(str_list[3])
        delay_label = node_id_delay[node_id]
        label_lat_array.append(lat_label)
        label_lon_array.append(lon_label)
        label_delay_array.append(delay_label)

        target_lat_label_dict[node_id] = lat_label
        target_lon_label_dict[node_id] = lon_label
        target_delay_label_dict[node_id] = delay_label

    #
    train_node_id_list = []  # 7:2:1
    lat_train_node_label_list = []
    lon_train_node_label_list = []
    delay_train_node_label_list = []

    val_node_id_list = []
    lat_val_node_label_list = []
    lon_val_node_label_list = []
    delay_val_node_label_list = []

    test_node_id_list = []
    lat_test_node_label_list = []
    lon_test_node_label_list = []
    delay_test_node_label_list = []


    for key, item in target_lat_label_dict.items():
        node_id = int(key)
        lat_label = target_lat_label_dict[node_id]
        lon_label = target_lon_label_dict[node_id]
        delay_label = target_delay_label_dict[node_id]

        rd_number = rd.random()
        if (rd_number < 0.1):
            test_node_id_list.append(node_id)
            lat_test_node_label_list.append(lat_label)
            lon_test_node_label_list.append(lon_label)
            delay_test_node_label_list.append(delay_label)

        elif (rd_number < 0.3) & (rd_number >= 0.1):
            val_node_id_list.append(node_id)
            lat_val_node_label_list.append(lat_label)
            lon_val_node_label_list.append(lon_label)
            delay_val_node_label_list.append(delay_label)

        else:
            train_node_id_list.append(node_id)
            lat_train_node_label_list.append(lat_label)
            lon_train_node_label_list.append(lon_label)
            delay_train_node_label_list.append(delay_label)

    delay_train_node_id_list=train_node_id_list.copy()#
    for key, item in node_id_delay.items():
        node_id = int(key)
        delay_label = item
        if(node_id not in test_node_id_list) & (node_id not in train_node_id_list) & (node_id not in val_node_id_list):
            delay_train_node_id_list.append(node_id)
            delay_train_node_label_list.append(delay_label)

    train_labeled_nodes = torch.tensor(train_node_id_list)
    lat_train_labels = torch.tensor(lat_train_node_label_list)  # their labels are different
    lon_train_labels = torch.tensor(lon_train_node_label_list)  # their labels are different
    delay_train_labels = torch.tensor(delay_train_node_label_list)
    delay_train_labeled_nodes = torch.tensor(delay_train_node_id_list)

    val_labeled_nodes = torch.tensor(val_node_id_list)
    lat_val_labels = torch.tensor(lat_val_node_label_list)  # their labels are different
    lon_val_labels = torch.tensor(lon_val_node_label_list)  # their labels are different
    delay_val_labels = torch.tensor(delay_val_node_label_list)

    test_labeled_nodes = torch.tensor(test_node_id_list)
    lat_test_labels = torch.tensor(lat_test_node_label_list)  # their labels are different
    lon_test_labels = torch.tensor(lon_test_node_label_list)  # their labels are different
    delay_test_labels = torch.tensor(delay_test_node_label_list)

    new_label_lat_array = target_scaler1.fit_transform(
        np.array(lat_train_node_label_list).reshape(-1, 1))
    new_label_lon_array = target_scaler2.fit_transform(np.array(lon_train_node_label_list).reshape(-1, 1))  

    new_label_lat_array = new_label_lat_array.astype(np.float32)  # RuntimeError: Found dtype Double but expected Float
    new_label_lon_array = new_label_lon_array.astype(np.float32)  # RuntimeError: Found dtype Double but expected Float

    new_lat_train_labels = torch.tensor(new_label_lat_array)
    new_lon_train_labels = torch.tensor(new_label_lon_array)

    #方便计算loss 获取val的实际loss
    new_lat_val_array = target_scaler1.transform(np.array(lat_val_node_label_list).reshape(-1, 1))
    new_lon_val_array= target_scaler2.transform(np.array(lon_val_node_label_list).reshape(-1, 1))
    new_lat_test_array = target_scaler1.transform(np.array(lat_test_node_label_list).reshape(-1, 1))
    new_lon_test_array = target_scaler2.transform(np.array(lon_test_node_label_list).reshape(-1, 1))
    new_lat_val_array = new_lat_val_array.astype(np.float32)  # RuntimeError: Found dtype Double but expected Float
    new_lon_val_array = new_lon_val_array.astype(np.float32)  # RuntimeError: Found dtype Double but expected Float
    new_lat_test_array = new_lat_test_array.astype(np.float32)  # RuntimeError: Found dtype Double but expected Float
    new_lon_test_array = new_lon_test_array.astype(np.float32)  # RuntimeError: Found dtype Double but expected Float
    new_lat_val_labels = torch.tensor(new_lat_val_array)
    new_lon_val_labels = torch.tensor(new_lon_val_array)
    new_lat_test_labels = torch.tensor(new_lat_test_array)
    new_lon_test_labels = torch.tensor(new_lon_test_array)

    return [train_labeled_nodes, new_lat_train_labels, new_lon_train_labels, lat_train_labels, lon_train_labels, delay_train_labeled_nodes, delay_train_labels,
            val_labeled_nodes, new_lat_val_labels,new_lon_val_labels,lat_val_labels, lon_val_labels, delay_val_labels,
            test_labeled_nodes, new_lat_test_labels,new_lon_test_labels,lat_test_labels, lon_test_labels , delay_test_labels]

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=3):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']
    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step,
                                                                    best_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def mediannum(num):
    listnum = [num[i] for i in range(len(num))]
    listnum.sort()
    lnum = len(num)
    if lnum % 2 == 1:
        i = int((lnum + 1) / 2) - 1
        return listnum[i]
    else:
        i = int(lnum / 2) - 1
        return (listnum[i] + listnum[i + 1]) / 2

if __name__ == '__main__':
    # -------check cuda GPU

    use_cuda = args.gpuid >= 0 and torch.cuda.is_available()
    if use_cuda:
        # torch.cuda.set_device(args.gpuid)
        torch.cuda.set_device(0)
    print(use_cuda)

    # output----------------------------------
    filepath = os.path.split(os.path.realpath(__file__))[0]
    from time import time
    stamp = int(time())
    result_path1 = filepath + '/output_mgnn_u_hk_/' + str(stamp) + '_loss.txt'  # 

    if not os.path.exists("./output_mgnn_u_hk_"):
        os.makedirs("./output_mgnn_u_hk_/")

    fw1 = open(result_path1, 'w')
    perf_str = 'aggregator_type=%s,gnn_dropout=%s,edge_dropout=%s,nn_dropout1=%s,nn_dropout2=%s,regs=%s,node_embedding_size=%s, node_hidden_size=%s,edge_hidden_size=%s,fc_size=%s,num_step_message_passing=%s, loss1_weight=%s,lr=%.4f\n' \
               % (aggregator_type, args.gnn_dropout, args.edge_dropout, args.nn_dropout1, args.nn_dropout2, args.regs,
                  args.node_embedding_size, args.node_hidden_size, args.edge_hidden_size, args.fc_size,
                  args.num_step_message_passing, loss1_weight, args.lr)

    fw1.write(perf_str)
    if print_flag > 0:
        print(perf_str)
    fw1.flush()

    loss_loger, error_loger = [], []

    stopping_step = 0

    # DATALOADER

    G, edge_attr_array = build_graph2(filepath + '/data/hk_edge_feature_median.txt')  #
    node_attr_array = get_node_attr2(filepath + '/data/hk_ip_feature_median.txt')  #

    if print_flag > 0:
        print('We have %d nodes.' % G.number_of_nodes())
        print('We have %d edges.' % G.number_of_edges())
    embed = nn.Embedding(G.number_of_nodes(), node_embedding_size)  # 
    G.ndata['feat'] = embed.weight  # 

    node_number = G.number_of_nodes()

    from model import MPNN
    net = MPNN(aggregator_type, node_embedding_size + 5, node_hidden_size, 10, edge_hidden_size,
               num_step_message_passing, gnn_dropout, edge_dropout,
               nn_dropout1)

    # MODEL GPU CUDA
    if use_cuda:
        net.cuda()

    inputs = embed.weight

    train_labeled_nodes, new_lat_train_labels, new_lon_train_labels, lat_train_labels, lon_train_labels, delay_train_labeled_nodes, delay_train_labels,\
    val_labeled_nodes, new_lat_val_labels, new_lon_val_labels, lat_val_labels, lon_val_labels, delay_val_labels,\
    test_labeled_nodes, new_lat_test_labels, new_lon_test_labels, lat_test_labels, lon_test_labels, delay_test_labels=\
        train_test_val_new_targetscaler2(filepath + '/data_0330/' + "hk_dstip_id_allinfo.txt")

    optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=lr,
                                 weight_decay=regs)  # regs float

    all_logits = []

    loss_fn = nn.MSELoss()
    if use_cuda:
        loss_fn = loss_fn.cuda()
    # node_feat = np.hstack((inputs,node_attr_array))
    node_attr_tensor = torch.from_numpy(node_attr_array).to(torch.float32)
    edge_attr_tensor = torch.from_numpy(edge_attr_array).to(torch.float32)
    node_feat = torch.cat([inputs, node_attr_tensor], dim=1)
    # 
    if use_cuda:
        node_feat = node_feat.cuda()
        edge_attr_tensor = edge_attr_tensor.cuda()  #
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        G = G.to(device)

        train_labeled_nodes_cpu = train_labeled_nodes
        val_labeled_nodes_cpu = val_labeled_nodes
        test_labeled_nodes_cpu = test_labeled_nodes
        delay_train_labeled_nodes_cpu = delay_train_labeled_nodes

        train_labeled_nodes, new_lat_train_labels, new_lon_train_labels, lat_train_labels, lon_train_labels, delay_train_labeled_nodes, delay_train_labels, \
        val_labeled_nodes, new_lat_val_labels, new_lon_val_labels, lat_val_labels, lon_val_labels, delay_val_labels, \
        test_labeled_nodes, new_lat_test_labels, new_lon_test_labels, lat_test_labels, lon_test_labels, delay_test_labels = \
            train_labeled_nodes.cuda(), new_lat_train_labels.cuda(), new_lon_train_labels.cuda(), lat_train_labels.cuda(), lon_train_labels.cuda(), delay_train_labeled_nodes.cuda(), delay_train_labels.cuda(), \
            val_labeled_nodes.cuda(), new_lat_val_labels.cuda(), new_lon_val_labels.cuda(), lat_val_labels.cuda(), lon_val_labels.cuda(), delay_val_labels.cuda(), \
            test_labeled_nodes.cuda(), new_lat_test_labels.cuda(), new_lon_test_labels.cuda(), lat_test_labels.cuda(), lon_test_labels.cuda(), delay_test_labels.cuda()
    cur_best_pre_0 = np.inf

    for epoch in range(epochs):
        net.train()
        logits,logits2 = net(G, node_feat, edge_attr_tensor)  #
        y1_predict = logits[:, 0]
        y2_predict = logits[:, 1]
        y3_predict = logits2[:, 0]

        y1_train_loss = loss_fn(y1_predict[train_labeled_nodes], new_lat_train_labels.squeeze(-1))
        y2_train_loss = loss_fn(y2_predict[train_labeled_nodes], new_lon_train_labels.squeeze(-1))
        y3_train_loss = loss_fn(y3_predict[delay_train_labeled_nodes], delay_train_labels.squeeze(-1))#delay_train_labeled_nodes

        total_loss = y1_train_loss + y2_train_loss + loss1_weight *  y3_train_loss

        temp_y1_predict = y1_predict.detach().cpu().numpy()
        temp_y2_predict = y2_predict.detach().cpu().numpy()
        temp_y3_predict = y3_predict.detach().cpu().numpy()
        old_y1_predict = target_scaler1.inverse_transform(temp_y1_predict.reshape(-1, 1))
        old_y2_predict = target_scaler2.inverse_transform(temp_y2_predict.reshape(-1, 1))
        # if 1:#(epoch%10==1):  # : only used in early model design
        #
        #
        #     sum_dis = 0
        #     sum_delay_error = 0
        #     for i, temp in enumerate(lat_train_labels):
        #         # temp_dis = 0
        #         temp_lat_label = temp.item()
        #         temp_lon_label = lon_train_labels[i].item()
        #
        #         temp_lat_pre = old_y1_predict[train_labeled_nodes_cpu][i].item()
        #         temp_lon_pre = old_y2_predict[train_labeled_nodes_cpu][i].item()
        #
        #         temp_dis = geodistance(temp_lon_label, temp_lat_label, temp_lon_pre, temp_lat_pre)
        #         sum_dis += temp_dis
        #
        #         temp_delay_label = delay_train_labels[i].item()
        #         temp_delay_pre = temp_y3_predict[i]
        #         sum_delay_error += abs(temp_delay_pre-temp_delay_label)
        #     error_thisepoch = sum_dis / len(lat_train_labels)
        #     error_thisepoch_delay = sum_delay_error/ len(lat_train_labels)
        #     perf_str = 'Train Epoch %d | Totalloss  %.4f  = Loss1 %.4f +  Loss2: %.4f +  Loss2: %.4f | avg_error=%.3f, avg_error_delay=%.3f' % (
        #         epoch, total_loss.item(), y1_train_loss.item(), y2_train_loss.item(),y3_train_loss.item(), error_thisepoch,error_thisepoch_delay)
        #     if print_flag > 0:
        #         print(perf_str)
        #-----------------------

        optimizer.zero_grad()  # 
        total_loss.backward()
        optimizer.step()

        net.eval()


        with torch.no_grad():
            if use_cuda:
                net.cpu()

            if 1:#:(epoch%10==1):  #
                y1_val_loss = loss_fn(y1_predict[val_labeled_nodes], new_lat_val_labels.squeeze(-1))  # loss
                y2_val_loss = loss_fn(y2_predict[val_labeled_nodes], new_lon_val_labels.squeeze(-1))
                y3_val_loss = loss_fn(y3_predict[val_labeled_nodes], delay_val_labels.squeeze(-1))
                total_val_loss = y1_val_loss + y2_val_loss + loss1_weight *  y3_val_loss
                sum_dis = 0
                sum_delay_error = 0
                for i, temp in enumerate(lat_val_labels):
                    # temp_dis = 0
                    temp_lat_label = temp.item()
                    temp_lon_label = lon_val_labels[i].item()

                    temp_lat_pre = old_y1_predict[val_labeled_nodes_cpu][i].item()
                    temp_lon_pre = old_y2_predict[val_labeled_nodes_cpu][i].item()
                    temp_dis = geodistance(temp_lon_label, temp_lat_label, temp_lon_pre, temp_lat_pre)

                    sum_dis += temp_dis

                    temp_delay_label = delay_val_labels[i].item()
                    temp_delay_pre = temp_y3_predict[i]
                    sum_delay_error += abs(temp_delay_pre - temp_delay_label)

                error_thisepoch = sum_dis / len(lat_val_labels)
                error_thisepoch_delay = sum_delay_error / len(lat_train_labels)

                loss_loger.append(total_val_loss.item())
                error_loger.append(error_thisepoch)

                perf_str = 'Val   Epoch %d | Totalloss  %.4f  = Loss1 %.4f +  Loss2: %.4f +  Loss2: %.4f | avg_error=%.3f, avg_error_delay=%.3f' % (
                    epoch, total_val_loss.item(), y1_val_loss.item(), y2_val_loss.item(), y3_val_loss.item(),
                    error_thisepoch, error_thisepoch_delay)

                if print_flag > 0:
                    print(perf_str)
                fw1.write(perf_str + '\n')
                fw1.flush()
                cur_best_pre_0, stopping_step, should_stop = early_stopping(error_thisepoch, cur_best_pre_0,
                                                                            stopping_step, expected_order='dec',
                                                                            flag_step=1000)#
                # ---updater
                y3_predict_copy = y3_predict.clone()

                y3_predict_copy[delay_train_labeled_nodes] = delay_train_labels.squeeze(-1)

                y3_predict_copy[val_labeled_nodes] = delay_val_labels.squeeze(-1)


                index_list = [[0]] * node_number#IP总数，2530
                y3_predict_copy = y3_predict_copy.view(node_number, 1)
                index = torch.LongTensor(index_list)
                index = index.cuda()

                node_feat = node_feat.scatter_(1, index, y3_predict_copy)  # , [2, 0, 0, 1, 2]
                # ---------------
                if use_cuda:
                    net.cuda()
                # *********************************************************
                # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
                if should_stop == True:
                    break
            # test process!!!!!only used at last when finishing tuning parameters
            # if epoch == args.epochs -1 :
            #
            #     y1_test_loss = loss_fn(y1_predict[test_labeled_nodes], lat_test_labels)  # 计算loss
            #     y2_test_loss = loss_fn(y2_predict[test_labeled_nodes], lon_test_labels)
            #     total_test_loss = loss1_weight * y1_test_loss + y2_test_loss
            #
            #     sum_dis = 0
            #     dis_list=[]
            #     for i, temp in enumerate(lat_test_labels):
            #         # temp_dis = 0
            #         temp_lat_label = temp.item()
            #         temp_lon_label = lon_test_labels[i].item()
            #         temp_lat_pre = old_y1_predict[test_labeled_nodes_cpu][i].item()
            #         temp_lon_pre = old_y2_predict[test_labeled_nodes_cpu][i].item()
            #         temp_dis = geodistance(temp_lon_label, temp_lat_label, temp_lon_pre, temp_lat_pre)
            #
            #         sum_dis += temp_dis
            #         dis_list.append(temp_dis)
            #     error_thisepoch = sum_dis / len(lat_test_labels)
            #     mediannum_dis = mediannum(dis_list)
            #     max_dis = max(dis_list)
            #     # perf_str = 'Val Epoch %d : val_loss==[%.5f], avg_error=%.5f' %  (epoch, total_val_loss.item(), error_thisepoch)
            #     perf_str = 'Testloss  %.4f  = Loss1 %.4f +  Loss2: %.4f | avg_error=%.5f, mediannum_dis=%.5f,max=%.5f,' % (
            #         total_test_loss.item(), y1_test_loss.item(), y2_test_loss.item(), error_thisepoch, mediannum_dis,max_dis)
            #     if print_flag > 0:
            #         print(perf_str)
            #     # fw3.write(perf_str + '\n')
            #     # fw3.flush()
            #     print(dis_list)
            #
            #     # 
            #     # val1 = old_y1_predict[val_labeled_nodes_cpu]
            #     # val2 = old_y2_predict[val_labeled_nodes_cpu]
            #     # test1 = old_y1_predict[test_labeled_nodes_cpu]
            #     # test2 = old_y2_predict[test_labeled_nodes_cpu]
            #     #
            #     # valdata = np.c_[val1, val2]
            #     # testdata = np.c_[test1, test2]
            #     #
            #     # real_valdata = np.c_[lat_val_labels, lon_val_labels]
            #     # real_testdata = np.c_[lat_test_labels, lon_test_labels]
            #     #
            #     # np.savetxt('data_nodelay_limit/hk_predict_val.txt', valdata)
            #     # np.savetxt('data_nodelay_limit/hk_predict_test.txt', testdata)
            #     #
            #     # np.savetxt('data_nodelay_limit/hk_real_val.txt', real_valdata)
            #     # np.savetxt('data_nodelay_limit/hk_real_test.txt', real_testdata)
            #     #
            #
            #     break
    # old code barely used, we directly read output loss files instead
    error_array = np.array(error_loger)

    best_error_0 = min(error_array)

    idx = list(error_array).index(best_error_0)

    final_perf = "Best Iter=@[%.1f]\terror=[%s]" % \
                 (idx, '\t'.join(['%.5f' % error_array[idx]]),
                  )
    if print_flag > 0:
        print(final_perf)

