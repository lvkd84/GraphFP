import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        
    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(5, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(3, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        if edge_attr != None:
            #add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

            edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        else:
            edge_embeddings = torch.zeros((edge_index[0].shape[1],x.shape[-1])).to(x.device)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GIN(torch.nn.Module):
    """
    
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0, atom = True):
        super(GIN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if atom:
            self.x_embedding1 = torch.nn.Embedding(120, emb_dim)
            self.x_embedding2 = torch.nn.Embedding(3, emb_dim)

            torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        else:
            self.embedding = nn.Embedding(800,emb_dim)

        self.atom = atom

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        if self.atom:
            x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        else:
            x = self.embedding(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        node_representation = h_list[-1]

        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self,
                 num_tasks,
                 emb_dim = 300,
                 num_gnn_layers = 5,
                 dropout=0.0,
                 graph_pooling="mean",
                 atom = True):
        super(GNN_graphpred, self).__init__()

        self.gnn = GIN(num_layer = num_gnn_layers,
                       emb_dim = emb_dim,
                       drop_ratio = dropout,
                       atom = atom)
        
        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        
        self.graph_pred = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(emb_dim,num_tasks)
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x,batch)

        return self.graph_pred(x)

from torch_geometric.data import DataLoader
# from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from tqdm import tqdm
# from ogb.graphproppred import Evaluator

# from loader import MoleculeDataset
from prepare_data_old import MoleculeDownstreamDataset
from splitters import scaffold_split

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downstream training')
    parser.add_argument('--dataset', type=str, required=True, help='downstream dataset name')
    parser.add_argument('--pretrain_path', type=str, default=None, help = "Pretrain model's path.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--epoch', type=int, default=100, help = "Number of training epochs.")
    parser.add_argument('--batchsize', type=int, default=256, help = "Batch size.")
    parser.add_argument('--dim', type=int, default=300, help = "Number of hidden dimensions.")
    parser.add_argument('--layers', type=int, default=5, help = "Number of mol gnn layers.")
    parser.add_argument('--lr', type=float, default=0.001, help = "Learning rate.")
    parser.add_argument('--drop', type=float, default=0, help = "Dropout rate.")
    parser.add_argument('--device', type=int, default=0, help = "GPU.")
    args = parser.parse_args()
    
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)

    device = torch.device("cuda:"+str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    dataset_name = args.dataset

    print("DATASET:", dataset_name)

    dataset = MoleculeDownstreamDataset("chem_dataset/{}".format(dataset_name), dataset = dataset_name)

    smiles_list = pd.read_csv('chem_dataset/{}/processed/smiles.csv'.format(dataset_name), header=None)[0].tolist()

    train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)

    num_tasks = train_dataset[0].y.shape[-1]

    trainloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers = 8)

    valloader = DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False, num_workers = 8)

    testloader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers = 8)

    model = GNN_graphpred(num_tasks, emb_dim=args.dim, num_gnn_layers=args.layers, dropout = args.drop, graph_pooling = 'mean')
    if args.pretrain_path:
        check_points = torch.load(args.pretrain_path, map_location=device)
        if 'gnn' in check_points.keys():
            model.gnn.load_state_dict(check_points['gnn'])
        else:
            model.gnn.load_state_dict(check_points['mol_gnn'])
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(reduction = "none")
    optimizer = Adam(model.parameters(),
                    lr=args.lr,
                    weight_decay=0)

    scheduler = StepLR(optimizer, step_size=30, gamma=0.3)

    best_val = 0
    best_test = None

    for epoch in tqdm(range(args.epoch)):
        model.train()
        cum_loss = 0
        for step, batch in enumerate(trainloader):

            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            y = batch.y.view(pred.shape).to(torch.float64)

            #Whether y is non-null or not.
            is_valid = y**2 > 0
            #Loss matrix
            loss_mat = criterion(pred.double(), (y+1)/2)
            #loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                
            optimizer.zero_grad()
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            loss.backward()

            optimizer.step()

            cum_loss += float(loss.cpu().item())

        scheduler.step()

        # VAL
        model.eval()
        y_pred = []
        y_true = []
        for step, batch in enumerate(valloader):
            # batch.y[batch.y == -1] = 0
            batch = batch.to(device)
            with torch.no_grad():
                pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)

            y_true.append(batch['y'].view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim = 0).numpy()
        y_pred = torch.cat(y_pred, dim = 0).numpy()

        roc_list = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                is_valid = y_true[:,i]**2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_pred[is_valid,i]))

        val_res = sum(roc_list)/len(roc_list)

        # TEST
        model.eval()
        y_pred = []
        y_true = []
        for step, batch in enumerate(testloader):
            # batch.y[batch.y == -1] = 0
            batch = batch.to(device)
            with torch.no_grad():
                pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)

            y_true.append(batch['y'].view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim = 0).numpy()
        y_pred = torch.cat(y_pred, dim = 0).numpy()

        roc_list = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                is_valid = y_true[:,i]**2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_pred[is_valid,i]))

        test_res = sum(roc_list)/len(roc_list)

        if val_res > best_val:
            best_val = val_res
            best_test = test_res

        # print("Val Res:", val_res, " Test Res:", test_res)

    print("Best Val:", best_val, " Best Test:", best_test)
