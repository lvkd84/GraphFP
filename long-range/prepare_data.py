import torch
import networkx as nx

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
}

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

class OGBAtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, full_atom_feature_dims):
        super(OGBAtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding

class OGBBondEncoder(torch.nn.Module):
    def __init__(self, emb_dim, full_bond_feature_dims):
        super(OGBBondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding 

class OGBFeaturizer:

    def __init__(self):
        super()

    def atom_to_feature_vector(self,atom):
        """
        Converts rdkit atom object to feature list of indices
        :param mol: rdkit atom object
        :return: list
        """
        atom_feature = [
                safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
                allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
                safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
                safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
                safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
                safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
                safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
                allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
                allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
                ]
        return atom_feature

    def bond_to_feature_vector(self,bond):
        """
        Converts rdkit bond object to feature list of indices
        :param mol: rdkit bond object
        :return: list
        """
        bond_feature = [
                    safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                    allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                    allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
                ]
        return bond_feature

    @classmethod
    def get_atom_encoder(cls,emb_dim):
        return OGBAtomEncoder(emb_dim, cls.get_atom_feature_dims())

    @classmethod
    def get_bond_encoder(cls,emb_dim):
        return OGBBondEncoder(emb_dim, cls.get_bond_feature_dims())
    
    @staticmethod
    def get_atom_feature_dims():
        return list(map(len, [
            allowable_features['possible_atomic_num_list'],
            allowable_features['possible_chirality_list'],
            allowable_features['possible_degree_list'],
            allowable_features['possible_formal_charge_list'],
            allowable_features['possible_numH_list'],
            allowable_features['possible_number_radical_e_list'],
            allowable_features['possible_hybridization_list'],
            allowable_features['possible_is_aromatic_list'],
            allowable_features['possible_is_in_ring_list']
        ]))
    
    @staticmethod
    def get_bond_feature_dims():
        return list(map(len, [
            allowable_features['possible_bond_type_list'],
            allowable_features['possible_bond_stereo_list'],
            allowable_features['possible_is_conjugated_list']
        ]))

from rdkit import Chem
import numpy as np

def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    featurizer = OGBFeaturizer()

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(featurizer.atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = featurizer.bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph

import os
import os.path as osp
import shutil
import gzip
import pandas as pd

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from networkx import weisfeiler_lehman_graph_hash
from tqdm import tqdm
from mol_bpe import Tokenizer


class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, smiles_column=None, data_file_path=None, vocab_file_path=None):
        
        self.smiles_column = smiles_column
        self.data_file_path = data_file_path
        self.vocab_file_path = vocab_file_path
        if self.vocab_file_path:
            self.tokenizer = Tokenizer(vocab_file_path)
            self.vocab_dict = {smiles:i for i,smiles in enumerate(self.tokenizer.vocab_dict.keys())}
        self.folder = root

        super(MoleculeDataset, self).__init__(self.folder, transform = None, pre_transform = None)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        data_df = pd.read_csv(self.data_file_path)

        if not (self.smiles_column in data_df.columns):
            raise ValueError("The specified SMILES column name is not found in the data file.")

        if data_df.isnull().values.any():
            raise ValueError("Missing values found in the data file.")

        tasks = [column for column in data_df.columns if column != self.smiles_column]
        smiles_list = data_df[self.smiles_column]
        task_list = data_df[tasks]

        data_list = []
        for i in tqdm(range(len(smiles_list))):
            
            data = Data()

            smiles = smiles_list[i]
            task_labels = task_list.iloc[i].values
            graph = smiles2graph(smiles)
            try:
                tree = self.tokenizer(smiles)
            except:
                print("Unable to process SMILES:", smiles)
                continue
            
            assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert(len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor(task_labels)#[None,:]

            # Manually consructing the fragment graph
            map = [0]*data.__num_nodes__
            frag = [[0] for _ in range(len(tree.nodes))]
            frag_edge_index = [[],[]]

            try:
                for node_i in tree.nodes:
                    node = tree.get_node(node_i)
                    # for atom in node, set map
                    for atom_i in node.atom_mapping.keys():
                        map[atom_i] = node_i
                    # extend frag
                        frag[node_i][0] = self.vocab_dict[node.smiles]
                for src, dst in tree.edges:
                    # extend edge index
                    frag_edge_index[0].extend([src,dst])
                    frag_edge_index[1].extend([dst,src])
            except KeyError as e:
                print("Error in matching subgraphs", e)
                continue

            unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
            frag_unique = torch.zeros(801).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

            data.map = torch.LongTensor(map)
            data.frag = torch.LongTensor(frag)
            data.frag_edge_index = torch.LongTensor(frag_edge_index)
            data.frag_unique = frag_unique

            data_list.append(data)

        tree_dict = {}
        hash_str_list = []
        for data in data_list:
            tree = Data()
            tree.x = data.frag
            tree.edge_index = data.frag_edge_index
            nx_graph = to_networkx(tree, to_undirected=True)
            hash_str = weisfeiler_lehman_graph_hash(nx_graph)
            if hash_str not in tree_dict:
                tree_dict[hash_str] = len(tree_dict)
            hash_str_list.append(hash_str)

        tree = []
        for hash_str in hash_str_list:
            tree.append(tree_dict[hash_str])

        for i, data in enumerate(data_list):
            data.tree = tree[i]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

def mol_frag_collate(data_list):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects.
    The assignment vector :obj:`batch` is created on the fly."""

    batch = Data()
    # keys follow node
    node_sum_keys = ["edge_index"]
    # keys follow frag
    frag_sum_keys = ["frag_edge_index", "map"]
    # no sum keys
    no_sum_keys = ["edge_attr",  
                   "x",
                   "frag",
                   "frag_unique"]

    for key in node_sum_keys + frag_sum_keys + no_sum_keys:
        batch[key] = []

    batch.y = []

    batch.node_batch_size = []
    batch.node_batch = []

    batch.frag_batch_size = []
    batch.frag_batch = []

    cumsum_node = 0
    i_node = 0

    cumsum_frag = 0
    i_frag = 0
    
    for data in data_list:
        num_nodes = data.x.shape[0]

        num_frags = data.frag.shape[0]
        
        batch.node_batch_size.append(num_nodes)

        batch.frag_batch_size.append(num_frags)

        batch.node_batch.append(torch.full((num_nodes, ), i_node, dtype=torch.long))

        batch.frag_batch.append(torch.full((num_frags, ), i_frag, dtype=torch.long))

        batch.y.append(data.y)

        for key in node_sum_keys:
            item = data[key]
            item = item + cumsum_node
            batch[key].append(item)

        for key in frag_sum_keys:
            item = data[key]
            item = item + cumsum_frag
            batch[key].append(item)
        
        for key in no_sum_keys:
            item = data[key]
            batch[key].append(item)

        cumsum_node += num_nodes
        i_node += 1

        cumsum_frag += num_frags
        i_frag += 1

    batch.x = torch.cat(batch.x, dim=0)
    batch.edge_index = torch.cat(batch.edge_index, dim=-1)
    batch.edge_attr = torch.cat(batch.edge_attr, dim=0)
    batch.frag = torch.cat(batch.frag, dim=0)
    batch.frag_edge_index = torch.cat(batch.frag_edge_index, dim=-1)
    batch.frag_unique = torch.cat(batch.frag_unique, dim=0)
    batch.map = torch.cat(batch.map, dim=-1)
    # for key in keys:
    #     batch[key] = torch.cat(
    #         batch[key], dim=batch.cat_dim(key))
    batch.node_batch = torch.cat(batch.node_batch, dim=-1)
    batch.node_batch_size = torch.tensor(batch.node_batch_size)
    batch.frag_batch = torch.cat(batch.frag_batch, dim=-1)
    batch.frag_batch_size = torch.tensor(batch.frag_batch_size)

    batch.y = torch.stack(batch.y)

    batch.tree = torch.LongTensor([data.tree for data in data_list])

    return batch.contiguous()

def cat_dim(self, key):
    return -1 if key == "edge_index" else 0

def cumsum(self, key, item):
    r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
    should be added up cumulatively before concatenated together.
    .. note::
        This method is for internal use only, and should only be overridden
        if the batch concatenation process is corrupted for a specific data
        attribute.
    """
    return key == "edge_index"

def downstream_mol_frag_collate(data_list):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects.
    The assignment vector :obj:`batch` is created on the fly."""

    batch = Data()
    # keys follow node
    node_sum_keys = ["edge_index"]
    # keys follow frag
    frag_sum_keys = ["frag_edge_index"]
    # no sum keys
    no_sum_keys = ["edge_attr",  
                   "x",
                   "frag"]

    for key in node_sum_keys + frag_sum_keys + no_sum_keys:
        batch[key] = []

    batch.y = []

    batch.node_batch_size = []
    batch.node_batch = []

    batch.frag_batch_size = []
    batch.frag_batch = []

    cumsum_node = 0
    i_node = 0

    cumsum_frag = 0
    i_frag = 0
    
    for data in data_list:
        num_nodes = data.x.shape[0]

        num_frags = data.frag.shape[0]
        
        batch.node_batch_size.append(num_nodes)

        batch.frag_batch_size.append(num_frags)

        batch.node_batch.append(torch.full((num_nodes, ), i_node, dtype=torch.long))

        batch.frag_batch.append(torch.full((num_frags, ), i_frag, dtype=torch.long))

        batch.y.append(data.y)

        for key in node_sum_keys:
            item = data[key]
            item = item + cumsum_node
            batch[key].append(item)

        for key in frag_sum_keys:
            item = data[key]
            item = item + cumsum_frag
            batch[key].append(item)
        
        for key in no_sum_keys:
            item = data[key]
            batch[key].append(item)

        cumsum_node += num_nodes
        i_node += 1

        cumsum_frag += num_frags
        i_frag += 1

    batch.x = torch.cat(batch.x, dim=0)
    batch.edge_index = torch.cat(batch.edge_index, dim=-1)
    batch.edge_attr = torch.cat(batch.edge_attr, dim=0)
    batch.frag = torch.cat(batch.frag, dim=0)
    batch.frag_edge_index = torch.cat(batch.frag_edge_index, dim=-1)
    # for key in keys:
    #     batch[key] = torch.cat(
    #         batch[key], dim=batch.cat_dim(key))
    batch.node_batch = torch.cat(batch.node_batch, dim=-1)
    batch.node_batch_size = torch.tensor(batch.node_batch_size)
    batch.frag_batch = torch.cat(batch.frag_batch, dim=-1)
    batch.frag_batch_size = torch.tensor(batch.frag_batch_size)

    batch.y = torch.stack(batch.y)

    return batch.contiguous()

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare molecular data')
    parser.add_argument('--root', type=str, required=True, help='Path to the data folder')
    parser.add_argument('--data_file_path', type=str, help='If creation, path to raw data')
    parser.add_argument('--smiles_column', type=str, help='Name of the colum containing smiles in the raw data table')
    parser.add_argument('--vocab_file_path', type=str, help='If creation, path to vocab file')
    args = parser.parse_args()

    dataset = MoleculeDataset(root=args.root,
                              smiles_column=args.smiles_column,
                              data_file_path=args.data_file_path,
                              vocab_file_path=args.vocab_file_path)