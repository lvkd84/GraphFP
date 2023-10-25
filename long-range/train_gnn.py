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
        out = self.mlp(x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
    
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out

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

class GraphEmbedding(torch.nn.Module):
    def __init__(self,
                 emb_dim = 300,
                 num_gnn_layers = 5,
                 dropout=0.0,
                 atom=True):
        super(GraphEmbedding, self).__init__()

        if atom:
            self.embedding = AtomEncoder(emb_dim = int(emb_dim))
        else:
            self.embedding = nn.Embedding(801,emb_dim)

        self.gnn = GIN_VN(num_layers = num_gnn_layers,
                          emb_dim = emb_dim,
                          drop_ratio = dropout,
                          residual = True)
        
        self.proj = nn.Sequential(
            nn.Linear(emb_dim,emb_dim)
        )

    def forward(self, x, edge_index, batch, edge_attr=None): 
        x = self.embedding(x)
        x = self.gnn(x, edge_index, edge_attr, batch)
        return self.proj(x)

from prepare_data import MoleculeDataset, mol_frag_collate
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from torch_scatter import scatter
from tqdm import tqdm
from info_nce import InfoNCE

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fragment-based Pretraining')
    parser.add_argument('--root', type=str, required=True, help='downstream dataset name')
    parser.add_argument('--epoch', type=int, default=100, help = "number of training epochs.")
    parser.add_argument('--batchsize', type=int, default=256, help = "batch size.")
    parser.add_argument('--dim', type=int, default=300, help = "number of hidden dimensions.")
    parser.add_argument('--lr', type=float, default=0.001, help = "learning rate.")
    parser.add_argument('--mol_layers', type=int, default=5, help = "number of mol gnn layers.")
    parser.add_argument('--frag_layers', type=int, default=2, help = "number of frag gnn layers.")
    parser.add_argument('--device', type=int, default=0, help = "GPU.")
    parser.add_argument('--save_path', type=str, default=None, help='path to save pretrained checkpoints')
    args = parser.parse_args()

    dataset = MoleculeDataset(root=args.root)

    loader = DataLoader(dataset, batch_size=args.batchsize, collate_fn=mol_frag_collate)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    mol_model = GraphEmbedding(emb_dim=args.dim, num_gnn_layers=args.mol_layers, atom=True)
    frag_model = GraphEmbedding(emb_dim=args.dim, num_gnn_layers=args.frag_layers, atom=False)

    mol_model.to(device)
    frag_model.to(device)

    criterion = InfoNCE()
    optimizer = AdamW(list(mol_model.parameters()) + list(frag_model.parameters()),
                    lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer,
                                patience = 5)

    for epoch in range(args.epoch):

        print("EPOCH ", epoch)
        
        mol_model.train()
        frag_model.train()

        cum_loss = 0

        for step, batch in enumerate(tqdm(loader, desc="Train")):

            batch = batch.to(device)

            atom_embedding = mol_model(batch.x, batch.edge_index, batch.node_batch, batch.edge_attr)

            atom_sum_embedding = scatter(atom_embedding, batch.map, dim=0, reduce="mean")

            frag_embedding = frag_model(batch.frag.squeeze(), batch.frag_edge_index, batch.frag_batch)

            loss = criterion(atom_sum_embedding, frag_embedding)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cum_loss += float(loss.cpu().item())

        scheduler.step(cum_loss)
        print("LOSS ", cum_loss/step)

        if args.save_path:
            if (epoch+1) % 10 == 0:
                torch.save({
                    'mol_gnn':mol_model.gnn.state_dict(),
                    'atom_embedding':mol_model.embedding.state_dict(),
                    'frag_gnn':frag_model.gnn.state_dict(),
                    'frag_embedding':frag_model.embedding.state_dict()
                }, args.save_path)



