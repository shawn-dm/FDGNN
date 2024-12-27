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
from torch_geometric.utils import negative_sampling,dropout_adj, subgraph
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
import torch_geometric.transforms as T
from PID import PIDControl
from layernorm import LayerNorm
from adddegree import OneHotDegree
from torch_geometric.utils import degree
import time
from VGAE import VGAE

def ignore_warnings():
    warnings.filterwarnings("ignore")

ignore_warnings()


# Define MLP classifier
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



def evaluate(model, classifier,data):
    model.eval()
    classifier.eval()

    with torch.no_grad():
        x,_ = model(data.x, data.edge_index)
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




def run(data, args):
    pbar = tqdm(range(args.runs), unit='run')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = data.to(device)



    acc, f1, auc_roc, parity, equality,ts = np.zeros(args.runs), np.zeros(
        args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs),np.zeros(args.runs)

    for count in pbar:
        seed_everything(count + args.seed)
        start_time = time.time()
        best_val_tradeoff = 0
        in_channels = data.x.size(1)
        hidden_channels = args.hidden
        if (args.encoder == 'GCN'):
            model = GCNEncoder(in_channels, hidden_channels).to(device)
        elif (args.encoder == 'SAGE'):
            model = SAGEencoder(in_channels, hidden_channels).to(device)
        elif (args.encoder == 'APPNP'):
            model = APPNPencoder(in_channels, hidden_channels).to(device)


        model.reset_parameters()

        classm = MLPClassifier(hidden_channels, args.num_classes).to(device)
        classm.reset_parameters()

        optimizer = optim.Adam(model.parameters(), lr=args.lr)


        num_epochs = args.epochs
        for epoch in range(num_epochs):
            optimizer.zero_grad()


            rx,_ = model(data.x, data.edge_index)

            optimizer.step()

            classifier = classm.to(device)
            classifier.reset_parameters()
            classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.lr)
            classifier.train()
            classifier_optimizer.zero_grad()
            x, _ = model(data.x, data.edge_index)
            preds = classifier(x)
            mlp_loss = F.binary_cross_entropy_with_logits(
                preds[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(device))
            mlp_loss.backward()
            classifier_optimizer.step()

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

    return acc, f1, auc_roc, parity, equality, ts


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
    parser.add_argument('--seed', type=int, default=997)
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--ratio', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--scaling_factor', type=float, default=1)


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

