import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl.function as fn
from collections import OrderedDict
import numpy as np


class Embedding(nn.Module):
    def __init__(self, pretrain=None, embedding_size=100, device='cpu'):
        # pretrain is a dict mapping entities to numpy array (dense vector)
        super().__init__()
        self.entities = [i for i in pretrain]
        self.entity2id = {j : i for i, j in enumerate(self.entities)}
        n_entities = len(pretrain)
        pretrain_array = []
        for e in self.entities:
            pretrain_array.append(torch.tensor(pretrain[e]))
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(n_entities, embedding_size)
        self.embedding.weight = nn.Parameter(torch.stack(pretrain_array))
        self.device = device
        # nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        idx = [[[self.entity2id[sample[0]], self.entity2id[sample[-1]]] for sample in batch] for batch in x]
        return self.embedding(torch.LongTensor(idx).to(self.device))


class RGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, rel_names, n_conv):
        super().__init__()
        self.encoder = nn.ModuleList([dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_dim if i == 0 else hidden_dim, hidden_dim) for rel in rel_names}, aggregate='sum') for i in range(n_conv)])

    def forward(self, graph, x):
        h = x
        for conv in self.encoder:
            h = conv(graph, h)
            h = {k: F.relu(v) for k, v in h.items()}
        return h


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


class RelationMetaLearner(nn.Module):
    def __init__(self, few=5, embed_size=100, hidden_size=None, out_size=100, dropout=0.5):
        super().__init__()
        if hidden_size is None:
            hidden_size = [500, 200]
        hidden_size = [2 * embed_size] + hidden_size
        self.out_size = out_size
        self.fc = nn.ModuleList([nn.Sequential(OrderedDict([
                                ('fc',   nn.Linear(hidden_size[i - 1], hidden_size[i])),
                                ('bn',   nn.BatchNorm1d(few)),
                                ('relu', nn.LeakyReLU()),
                                ('drop', nn.Dropout(dropout)),])) for i in range(1, len(hidden_size))])
        self.out_fc = nn.Sequential(OrderedDict([
                        ('fc', nn.Linear(hidden_size[-1], out_size)),
                        ('bn', nn.BatchNorm1d(few)),]))

    def forward(self, x):
        size = x.shape
        for fc in self.fc:
            x = fc(x)
        x = self.out_fc(x)
        x = torch.mean(x, 1)
        return x.view(size[0], 1, self.out_size)


def embed_scoring(node1, node2, relation):
    score = -torch.norm(node1 + node2 - relation, 2, -1)
    # score = torch.min(-torch.norm(node1 + relation - node2, 2, -1),
    #                   -torch.norm(node2 + relation - node2, 2, -1))
    return score


class Model(nn.Module):
    def __init__(self, pretrain_embed, few, in_features=100, dropout=0.5, device='cpu'):
        super().__init__()
        self.encoder = Embedding(pretrain_embed, embedding_size=in_features, device=device)
        # self.encoder = RGCN(in_features, in_features, rel_names, n_conv=n_conv)
        self.relation_learner = RelationMetaLearner(few=few, embed_size=in_features, hidden_size=[500, 200], out_size=100, dropout=dropout)
        self.mem = dict()

    def train_forward(self, support, query, negative):
        support = self.encoder(support)
        rel = self.relation_learner(torch.cat([support[:,:,0,:], support[:,:,1,:]], dim=2))
        query = self.encoder(query)
        node1 = query[:,:,0,:]
        node2 = query[:,:,1,:]
        pos_score = embed_scoring(node1, node2, rel)
        negative = self.encoder(negative)
        node1 = negative[:,:,0,:]
        node2 = negative[:,:,1,:]
        neg_score = embed_scoring(node1, node2, rel)
        return pos_score, neg_score

    def forward(self, support, query):
        support = self.encoder(support)
        rel = self.relation_learner(torch.cat([support[:,:,0,:], support[:,:,1,:]], dim=2))
        node1 = query[:,:,0,:]
        node2 = query[:,:,1,:]
        pos_score = embed_scoring(node1, node2, rel)
        return pos_score
