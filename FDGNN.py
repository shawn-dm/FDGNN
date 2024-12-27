from dataset import *
from model import *
from utils import *
from learn import *
import argparse
from tqdm import tqdm
from torch import tensor
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, InnerProductDecoder
# from torch_geometric.nn.norm import LayerNorm
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import dropout_adj,subgraph
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
import torch_geometric.transforms as T
from PID import PIDControl
from layernorm import LayerNorm
from adddegree import OneHotDegree
from torch_geometric.utils import degree
import time
from sklearn.model_selection import GridSearchCV
import pandas as pd
from VGAE import VGAE

def ignore_warnings():
    warnings.filterwarnings("ignore")

ignore_warnings()


class GCNClassifer(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNClassifer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = torch.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.conv2(x, edge_index)
        return x



# Define loss function

def _sim(z1,z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

def infonce(z1,z2):
        temp = lambda x: torch.exp(x / 0.5)
        refl_sim = temp(_sim(z1, z1))
        between_sim = temp(_sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def loss_function(rx,rx2,rx3, rs, fx, fs, data):

    edge_index = data.edge_index
    pos_samples = torch.stack([rx, fx,rx2,rx3], dim=1)
    neg_samples = negative_sampling(edge_index, num_neg_samples=pos_samples.size(0), method="sparse")
    neg_rx = rx[neg_samples[0]]
    neg_rs = rs[neg_samples[0]]
    neg_fs = fs[neg_samples[1]]
    neg_fx = fx[neg_samples[1]]

    neg_sim_rx_rs = infonce(neg_rx,neg_rs)
    neg_sim_rs_fs = infonce(neg_fs,neg_rs)
    neg_sim_fx_fs = infonce(neg_fx,neg_fs)
    #
    pos_sim = infonce(rx,fx)
    pos_sim2 = infonce(rx,rx2)
    pos_sim3 = infonce(rx,rx3)



    pos_loss = torch.log(torch.sigmoid(torch.cat([pos_sim, pos_sim2, pos_sim3]) + 1e-8)).mean()
    neg_loss = torch.log(1- torch.sigmoid(torch.cat([neg_sim_rx_rs, neg_sim_rs_fs, neg_sim_fx_fs])) + 1e-8).mean()


    return  pos_loss + neg_loss




def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    device = x.device
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def random_walk_subgraph(edge_index, edge_weight = None, batch_size= 1000, length= 10):
    num_nodes = edge_index.max().item() + 1

    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))

    start = torch.randint(0, num_nodes, size=(batch_size, ), dtype=torch.long).to(edge_index.device)
    node_idx = adj.random_walk(start.flatten(), length).view(-1)

    edge_index, edge_weight = subgraph(node_idx, edge_index, edge_weight)

    return edge_index, edge_weight


class MLPClassifier(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Linear(hidden_channels, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.fc.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


    def forward(self, x):
        # x = F.normalize(x, p=2, dim=1) * args.scaling_factor
        x = self.fc(x)
        x = F.relu(x)
        x = F.dropout(x,p=0.5)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x,p=0.5)
        x = self.fc2(x)
        return torch.sigmoid(x)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Linear(in_channels, hidden_channels)

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, x):
        x = self.fc(x)
        return x

def evaluate(model, classifier,data):
    model.eval()
    classifier.eval()

    with torch.no_grad():
        x = model.encode(data.x, data.edge_index)
        output = classifier(x)


    accs, auc_rocs, F1s, paritys, equalitys = {}, {}, {}, {}, {}

    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)

    accs['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    accs['test'] = pred_test.eq(
        data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

    F1s['val'] = f1_score(data.y[data.val_mask].cpu(
    ).numpy(), pred_val.cpu().numpy())

    F1s['test'] = f1_score(data.y[data.test_mask].cpu(
    ).numpy(), pred_test.cpu().numpy())

    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    paritys['val'], equalitys['val'] = fair_metric(pred_val.cpu().numpy(), data.y[data.val_mask].cpu(
    ).numpy(), data.sens[data.val_mask].cpu().numpy())

    paritys['test'], equalitys['test'] = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    ).numpy(), data.sens[data.test_mask].cpu().numpy())

    return accs, auc_rocs, F1s, paritys, equalitys






def train_vgae(model,model_s, data, optimizer,optimizer_s , num_epochs,args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    x = data.x.to(device)
    s = data.x[:, args.sens_idx].to(device)
    s = torch.unsqueeze(s, dim=1).to(device)

    edge_d,_ = dropout_adj(data.edge_index)
    edge_d = edge_d.to(device)
    edge_r,_ = random_walk_subgraph(data.edge_index)
    # x_d = drop_feature(x,0.5)


    flipped_x = x.clone().to(device)
    flipped_x[:, args.sens_idx] = 1 - flipped_x[:, args.sens_idx]  # Flip sensitive attribute
    flipped_s = flipped_x[:, args.sens_idx].to(device)
    flipped_s = torch.unsqueeze(flipped_s, dim=1).to(device)


    model.train()
    model_s.train()
    optimizer.zero_grad()
    optimizer_s.zero_grad()
    edge_index = data.edge_index.to(device)

    rx = model.encode(x, edge_index).to(device)
    rx2 = model.encode(x, edge_d).to(device)

    rx3 = model.encode(x,edge_r).to(device)
    # Sensitive features
    rs = model_s.encode(s, edge_index).to(device)

    # Flipped features
    fx = model.encode(flipped_x, edge_index)
    fs = model_s.encode(flipped_s,edge_index)

    rec_loss = 0.1 * model.recon_loss(x, edge_index).to(device)

    # loss = loss_function(rx,rx2,rx3, rs, fx, fs, data) + rec_loss
    loss = loss_function(rx,rx2,rx3, rs, fx, fs, data)

    loss.backward()
    optimizer.step()
    optimizer_s.step()

def train_classifier(model,classm, data,classm_optimizer,args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    classm_optimizer.zero_grad()
    classm.train()

    tx = model.encode(data.x, data.edge_index)
    preds = classm(tx)
    mlp_loss = F.binary_cross_entropy_with_logits(
        preds[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(device))
    mlp_loss.backward(retain_graph=True)
    classm_optimizer.step()

def run(data, args):
    pbar = tqdm(range(args.runs), unit='run')
    acc, f1, auc_roc, parity, equality,ts = np.zeros(args.runs), np.zeros(
        args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs),np.zeros(args.runs)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    max_deg = deg.max().item()
    transform = OneHotDegree(max_degree=max_deg, in_degree=True)
    data = transform(data)
    data = data.to(device)



    in_channels = data.x.size(1)
    hidden_channels = args.hidden

    for count in pbar:
        start_time = time.time()
        seed_everything(count + args.seed)
        best_val_tradeoff = 0
        if (args.encoder == 'GCN'):
            encoder = GCNEncoder(in_channels, hidden_channels)
            encoder_s = GCNEncoder(1, hidden_channels)
        elif (args.encoder == 'SAGE'):
            encoder = SAGEencoder(in_channels, hidden_channels)
            encoder_s = SAGEencoder(1, hidden_channels)
        elif (args.encoder == 'APPNP'):
            encoder = APPNPencoder(in_channels, hidden_channels)
            encoder_s = APPNPencoder(1, hidden_channels)

        model = VGAE(encoder)
        model_s = VGAE(encoder_s)
        encoder.reset_parameters()
        encoder_s.reset_parameters()
        classm = MLPClassifier(hidden_channels, 1).to(device)
        classm.reset_parameters()
        classm_optimizer = optim.Adam(classm.parameters(), lr=args.lr)

        model = model.to(device)
        model_s = model_s.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer_s = optim.Adam(model_s.parameters(), lr=args.lr)

        num_epochs = args.epochs
        for epoch in range(num_epochs):
            train_vgae(model,model_s, data, optimizer,optimizer_s , 1,args)


        for epoch in range(num_epochs):
            train_classifier(model,classm, data,classm_optimizer, args)

            accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate(model,classm,data)
            if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (
                    tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
                    test_acc = accs['test']
                    test_auc_roc = auc_rocs['test']
                    test_f1 = F1s['test']
                    test_parity, test_equality = tmp_parity['test'], tmp_equality['test']
                    best_val_tradeoff = auc_rocs['val'] + F1s['val'] + \
                                        accs['val'] - (tmp_parity['val'] + tmp_equality['val'])
        end_time = time.time()
        total_time = end_time - start_time
        ts[count] = total_time
        acc[count] = test_acc
        f1[count] = test_f1
        auc_roc[count] = test_auc_roc
        parity[count] = test_parity
        equality[count] = test_equality



    return acc, f1, auc_roc, parity, equality ,ts



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='german')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--prop', type=str, default='scatter')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--ratio', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--scaling_factor', type=float, default=1.5)


    args = parser.parse_args()
    data, args.sens_idx, args.corr_sens, args.corr_idx, args.x_min, args.x_max = get_dataset(
        args.dataset, args.top_k)
    args.num_features, args.num_classes = data.x.shape[1], len(
        data.y.unique()) - 1
    args.train_ratio, args.val_ratio = torch.tensor([
        (data.y[data.train_mask] == 0).sum(), (data.y[data.train_mask] == 1).sum()]), torch.tensor([
            (data.y[data.val_mask] == 0).sum(), (data.y[data.val_mask] == 1).sum()])
    args.train_ratio, args.val_ratio = torch.max(
        args.train_ratio) / args.train_ratio, torch.max(args.val_ratio) / args.val_ratio
    args.train_ratio, args.val_ratio = args.train_ratio[
        data.y[data.train_mask].long()], args.val_ratio[data.y[data.val_mask].long()]

    acc, f1, auc_roc, parity, equality,ts = run(data, args)



    print('======' + args.dataset + args.encoder + '======')
    # print('auc_roc:', np.mean(auc_roc) * 100, np.std(auc_roc) * 100)
    print('Acc:', np.mean(acc) * 100, np.std(acc) * 100)
    print('f1:', np.mean(f1) * 100, np.std(f1) * 100)
    print('parity:', np.mean(parity) * 100, np.std(parity) * 100)
    print('equality:', np.mean(equality) * 100, np.std(equality) * 100)
    print('times:', np.mean(ts) , np.std(ts) )

