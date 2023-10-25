import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn as nn
import torch.nn.functional as F

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from torch_geometric.utils import degree

class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        # self.eps = torch.nn.Parameter(torch.Tensor([0]))
        
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr != None:
            edge_embedding = self.bond_encoder(edge_attr)
        else:
            edge_embedding = torch.zeros((edge_index.shape[1],x.shape[-1])).to(x.device)
            # print(x.shape, edge_embedding.shape)
        out = self.mlp(x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
    
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out

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
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0):
        super(GIN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(emb_dim))
            # self.gnns.append(GCNConv(emb_dim, aggr = "add"))

        ###List of layernorms
        self.layer_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            # self.layer_norms.append(torch.nn.LayerNorm(emb_dim))
            self.layer_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.layer_norms[layer](h_list[layer])
            h = self.gnns[layer](h, edge_index, edge_attr)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            # h_list = [h.unsqueeze_(0) for h in h_list]
            h_list = [torch.unsqueeze(h,dim=0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            # h_list = [h.unsqueeze_(0) for h in h_list]
            h_list = [torch.unsqueeze(h,dim=0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation

### Virtual GNN to generate node embedding
class GIN_VN(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layers, emb_dim, drop_ratio = 0.5, residual = False):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GIN_VN, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layers):
            # self.convs.append(GINConv(emb_dim))
            self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers):
            if layer == self.num_layers - 1:
                self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), \
                                                        torch.nn.Linear(emb_dim, emb_dim)))                
            else:
                self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), \
                                                        torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))


    def forward(self, x, edge_index, edge_attr, batch):

        ### virtual node embeddings for graphs
        # virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [x]
        for layer in range(self.num_layers):
            ### add message from virtual nodes to graph nodes
            # h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
                ### add message from graph nodes to virtual nodes
            # virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
            ### transform virtual nodes using MLP

            # if self.residual:
            #     virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
            # else:
            #     virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
        
        return h_list[-1]

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

        if atom:
            self.embedding = AtomEncoder(emb_dim = int(emb_dim))
        else:
            self.embedding = nn.Embedding(801,emb_dim)

        self.gnn = GIN_VN(num_layers = num_gnn_layers,
                          emb_dim = emb_dim,
                          drop_ratio = dropout,
                          residual = True)
        
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
            nn.Dropout(dropout),
            nn.Linear(emb_dim,num_tasks)
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        
        x = self.embedding(x)
        x = self.gnn(x, edge_index, edge_attr, batch)
        x = self.pool(x,batch)

        return x #self.graph_pred(x)

# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
# from ogb.graphproppred import PygGraphPropPredDataset
from prepare_data import MoleculeDataset, downstream_mol_frag_collate
# from ogb.graphproppred import Evaluator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

import argparse
import pickle
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downstream training')
    parser.add_argument('--dataset', type=str, required=True, help='downstream dataset name')
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--epoch', type=int, default=100, help = "number of training epochs.")
    parser.add_argument('--batchsize', type=int, default=128, help = "batch size.")
    parser.add_argument('--dim', type=int, default=300, help = "number of hidden dimensions.")
    parser.add_argument('--lr', type=float, default=0.001, help = "learning rate.")
    parser.add_argument('--drop', type=float, default=0.0, help = "dropout rate.")
    parser.add_argument('--mol_layers', type=int, default=5, help = "number of mol gnn layers.")
    parser.add_argument('--frag_layers', type=int, default=5, help = "number of frag gnn layers.")
    parser.add_argument('--device', type=int, default=0, help = "GPU.")
    parser.add_argument('--save_path', type=str, default=None, help='path to save pretrained checkpoints')
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)

    dataset = MoleculeDataset(root="data/{}_frag".format(args.dataset))

    if args.dataset == 'peptide_cls':
        f = open('data/splits_random_stratified_peptide.pickle','rb')
        split_idx = pickle.load(f)
        f.close()
        num_tasks = 10
        criterion = nn.BCEWithLogitsLoss()
    elif args.dataset == 'peptide_reg':
        f = open('data/splits_random_stratified_peptide_structure.pickle','rb')
        split_idx = pickle.load(f)
        f.close()
        num_tasks = 11
        criterion = nn.L1Loss()
    else:
        raise ValueError

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if args.save_path:
        check_points = torch.load(args.save_path)

    trainloader = DataLoader(dataset[split_idx["train"]], batch_size=args.batchsize, shuffle=True, collate_fn=downstream_mol_frag_collate, num_workers = 8)

    valloader = DataLoader(dataset[split_idx["val"]], batch_size=args.batchsize, shuffle=False, collate_fn=downstream_mol_frag_collate, num_workers = 8)

    testloader = DataLoader(dataset[split_idx["test"]], batch_size=args.batchsize, shuffle=False, collate_fn=downstream_mol_frag_collate, num_workers = 8)  

    mol_model = GNN_graphpred(num_tasks, emb_dim=args.dim, num_gnn_layers=args.mol_layers, dropout = args.drop, graph_pooling = 'mean')
    if args.save_path:
        mol_model.gnn.load_state_dict(check_points['mol_gnn'])
        mol_model.embedding.load_state_dict(check_points['atom_embedding'])
    mol_model.to(device)

    frag_model = GNN_graphpred(num_tasks, emb_dim=args.dim, num_gnn_layers=args.frag_layers, dropout = args.drop, graph_pooling = 'mean', atom=False)
    if args.save_path:
        # frag_model.gnn.load_state_dict(check_points['mol_gnn'])
        frag_model.embedding.load_state_dict(check_points['frag_embedding'])
    frag_model.to(device)

    graph_pred = nn.Sequential(
        nn.Linear(args.dim*2,num_tasks)
    )
    graph_pred.to(device)

    optimizer = Adam(list(mol_model.parameters()) + list(frag_model.parameters()) + list(graph_pred.parameters()),
                    lr=args.lr)

    scheduler = ReduceLROnPlateau(optimizer, patience = 20, factor=0.5)

    if args.dataset == 'peptide_cls':
        best_val = 0
    elif args.dataset == 'peptide_reg':
        best_val = 100
    best_test = None

    for epoch in tqdm(range(args.epoch)):
        mol_model.train()
        frag_model.train()
        graph_pred.train()
        cum_loss = 0
        for step, batch in enumerate(trainloader):
            batch = batch.to(device)
            mol_emb = mol_model(batch.x, batch.edge_index, batch.node_batch, batch.edge_attr)
            frag_emb = frag_model(batch.frag.squeeze(), batch.frag_edge_index, batch.frag_batch)
            pred = graph_pred(torch.cat([mol_emb,frag_emb],dim=-1))

            is_labeled = batch.y.view(pred.shape) == batch.y.view(pred.shape)
            loss = criterion(pred.float()[is_labeled], batch['y'].view(pred.shape).float()[is_labeled])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cum_loss += float(loss.cpu().item())

        scheduler.step(cum_loss)

        # VAL
        mol_model.train()
        frag_model.train()
        graph_pred.train()
        y_pred = []
        y_true = []
        for step, batch in enumerate(valloader):
            batch = batch.to(device)
            with torch.no_grad():
                mol_emb = mol_model(batch.x, batch.edge_index, batch.node_batch, batch.edge_attr)
                frag_emb = frag_model(batch.frag.squeeze(), batch.frag_edge_index, batch.frag_batch)
                pred = graph_pred(torch.cat([mol_emb,frag_emb],dim=-1))

            y_true.append(batch['y'].view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim = 0).numpy()
        y_pred = torch.cat(y_pred, dim = 0).numpy()

        if args.dataset == 'peptide_cls':
            roc_list = []
            for i in range(y_true.shape[1]):
                #AUC is only defined when there is at least one positive data.
                if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                    roc_list.append(average_precision_score(y_true[:,i], y_pred[:,i]))

            val_res = sum(roc_list)/len(roc_list)
        elif args.dataset == 'peptide_reg':
            mse_list = []
            for i in range(y_true.shape[1]):
                mse_list.append(mean_absolute_error(y_true[:,i], y_pred[:,i]))

            val_res = sum(mse_list)/len(mse_list)

        # TEST
        mol_model.train()
        frag_model.train()
        graph_pred.train()
        y_pred = []
        y_true = []
        for step, batch in enumerate(testloader):
            batch = batch.to(device)
            with torch.no_grad():
                # pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                mol_emb = mol_model(batch.x, batch.edge_index, batch.node_batch, batch.edge_attr)
                frag_emb = frag_model(batch.frag.squeeze(), batch.frag_edge_index, batch.frag_batch)
                pred = graph_pred(torch.cat([mol_emb,frag_emb],dim=-1))

            y_true.append(batch['y'].view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim = 0).numpy()
        y_pred = torch.cat(y_pred, dim = 0).numpy()

        if args.dataset == 'peptide_cls':
            roc_list = []
            for i in range(y_true.shape[1]):
                #AUC is only defined when there is at least one positive data.
                if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                    roc_list.append(average_precision_score(y_true[:,i], y_pred[:,i]))

            test_res = sum(roc_list)/len(roc_list)
        elif args.dataset == 'peptide_reg':
            mse_list = []
            for i in range(y_true.shape[1]):
                mse_list.append(mean_absolute_error(y_true[:,i], y_pred[:,i]))

            test_res = sum(mse_list)/len(mse_list)

        if args.dataset == 'peptide_cls':
            if val_res > best_val:
                best_val = val_res
                best_test = test_res
        elif args.dataset == 'peptide_reg':
            if val_res < best_val:
                best_val = val_res
                best_test = test_res

        # print("Val Res:", val_res, " Test Res:", test_res)
        # if (epoch + 1) % 100 == 0:
        #     print("Best Val:", best_val, " Best Test:", best_test)

    print("Best Val:", best_val, " Best Test:", best_test)
