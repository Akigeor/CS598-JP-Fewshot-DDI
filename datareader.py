from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os

import tensorflow.compat.v1 as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics

import csv
import json
from collections import defaultdict

import dgl
import torch

def read_data(folder='data/'):
    # Read data
    # proteins, proteins_id, drugs, drugs_id, edge_types, edge_types_id, drug_drug, protein_protein, drug_protein
    node_embedding = True

    cnt_edge_type = defaultdict(int)
    drugs = set()
    proteins = set()

    # drug-gene
    with open(folder + 'bio-decagon-targets.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        headers = next(reader, None)
        for row in reader:
            drug, gene = row
            drugs.add(drug)
            proteins.add(gene)

    # drug-drug
    with open(folder + 'bio-decagon-combo.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        headers = next(reader, None)
        for row in reader:
            drug1, drug2, edge_type, edge_type_name = row
            cnt_edge_type[edge_type] += 1
            drugs.add(drug1)
            drugs.add(drug2)
    edge_types = set()
    for i in cnt_edge_type:
        if cnt_edge_type[i] >= 500:
            edge_types.add(i)
    edge_types = list(sorted(edge_types))

    # gene-gene
    with open(folder + 'bio-decagon-ppi.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        headers = next(reader, None)
        for row in reader:
            gene1, gene2 = row
            proteins.add(gene1)
            proteins.add(gene2)

    make_id = lambda x : {j : i for i, j in enumerate(x)}

    proteins = list(sorted(proteins))
    proteins_id = make_id(proteins)
    drugs = list(sorted(drugs))
    drugs_id = make_id(drugs)
    edge_types_id = make_id(edge_types)

    drug_protein = []
    drug_drug = [[] for i in range(len(edge_types))]
    protein_protein = []

    # drug-gene
    with open(folder + 'bio-decagon-targets.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        headers = next(reader, None)
        for row in reader:
            drug, gene = row
            drug_protein.append((drugs_id[drug], proteins_id[gene]))

    # drug-drug
    with open(folder + 'bio-decagon-combo.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        headers = next(reader, None)
        for row in reader:
            drug1, drug2, edge_type, edge_type_name = row
            if edge_type in edge_types_id:
                drug_drug[edge_types_id[edge_type]].append((drugs_id[drug1], drugs_id[drug2]))

    # gene-gene
    with open(folder + 'bio-decagon-ppi.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        headers = next(reader, None)
        for row in reader:
            gene1, gene2 = row
            protein_protein.append((proteins_id[gene1], proteins_id[gene2]))

    # gene-monoeffect
    monoeffect = set()
    drug2monoeffect = defaultdict(list)
    with open(folder + 'bio-decagon-mono.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        headers = next(reader, None)
        for row in reader:
            drug1, edge_type2, ename = row
            monoeffect.add(edge_type2)
            drug2monoeffect[drug1].append(edge_type2)
    monoeffect = sorted(list(monoeffect))
    monoeffect_id = {j: i for i, j in enumerate(monoeffect)}

    # # monoeffect-class
    # monoeffect2class = defaultdict(list)
    # eff_classes = set()
    # with open(folder + 'bio-decagon-effectcategories.csv', newline='') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',')
    #     headers = next(reader, None)
    #     for row in reader:
    #         effect, eff_name, eff_class = row
    #         eff_class = eff_class.replace(' ', '_')
    #         if effect in monoeffect:
    #             eff_classes.add(eff_class)
    #             monoeffect2class.append(eff_class)
    # eff_classes = sorted(list(eff_classes))
    # eff_classes_id = {j: i for i, j in enumerate(eff_classes)}

    flag_monoeffect = True
    # flag_monoeffect_class = True
    G = nx.Graph()

    # Build graph nodes
    entities = [(i, {'type': 'proteins'}) for i in proteins] + [(i, {'type': 'drugs'}) for i in drugs]
    if flag_monoeffect:
        entities += [(i, {'type': 'monoeffect'}) for i in monoeffect]
    # if flag_monoeffect_class:
    #     entities += [(i, {'type': 'disease_class'}) for i in eff_classes]
    G.add_nodes_from(entities)

    # Build graph edges
    edges = []
    # for rel in range(len(drug_drug)):
    #     for d1, d2 in drug_drug[rel]:
    #         edges.append((drugs[d1], drugs[d2], {'type': edge_types[rel]}))
    # dd_edges = len(edges)
    for p1, p2 in protein_protein:
        edges.append((proteins[p1], proteins[p2], {'type': 'ppi'}))
    for d1, p2 in drug_protein:
        edges.append((drugs[d1], proteins[p2], {'type': 'target'}))
    if flag_monoeffect:
        for d1 in drug2monoeffect:
            for e1 in drug2monoeffect[d1]:
                edges.append((d1, e1, {'type': 'monoeffect'}))
    # if flag_monoeffect_class:
    #     for e1 in monoeffect2class:
    #         for c1 in monoeffect2class[e1]:
    #             edges.append((e1, c1, {'type': 'disease_class'}))
    G.add_edges_from(edges)

    # Build dgl HIN
    edge_dict = dict()
    P1 = [p1 for p1, p2 in protein_protein]
    P2 = [p2 for p1, p2 in protein_protein]
    P1, P2 = torch.tensor(P1 + P2), torch.tensor(P2 + P1)
    edge_dict[('protein', 'ppi', 'protein')] = (P1, P2)
    D1 = torch.tensor([d1 for d1, p2 in drug_protein])
    P2 = torch.tensor([p2 for d1, p2 in drug_protein])
    edge_dict[('drug', 'has_target', 'protein')] = (D1, P2)
    edge_dict[('protein', 'target_of', 'drug')] = (P2, D1)
    if flag_monoeffect:
        D1 = []
        E1 = []
        for d1 in drug2monoeffect:
            for e1 in drug2monoeffect[d1]:
                D1.append(drugs_id[d1])
                E1.append(monoeffect_id[e1])
        D1 = torch.tensor(D1)
        E1 = torch.tensor(E1)
        edge_dict[('drug', 'has_monoeffect', 'monoeffect')] = (D1, E1)
        edge_dict[('monoeffect', 'belong_to', 'drug')] = (E1, D1)
    # if flag_monoeffect_class:
    #     E1 = []
    #     C1 = []
    #     for e1 in monoeffect2class:
    #         for c1 in monoeffect2class[e1]:
    #             E1.append(monoeffect_id[e1])
    #             C1.append(eff_classes_id[c1])
    #     E1 = torch.tensor(E1)
    #     C1 = torch.tensor(C1)
    #     edge_dict[('monoeffect', 'in_class', 'disease_class')] = (E1, C1)
    #     edge_dict[('disease_class', 'class_contains', 'monoeffect')] = (C1, E1)
    HG = dgl.heterograph(edge_dict)

    # Load or initialize node features
    if node_embedding:
        entity2embed = dict()
        ent2id = json.load(open(folder + 'ent2ids'))
        embed_array = np.load(folder + 'ent2vec.npy')
        feature_dim = embed_array.shape[1]
        for i in ent2id:
            entity2embed[i] = embed_array[ent2id[i]]
        HG.nodes['drug'].data['feature'] = torch.stack([torch.tensor(entity2embed[drugs[i]]) for i in HG.nodes('drug')])
        HG.nodes['protein'].data['feature'] = torch.stack([torch.tensor(entity2embed[proteins[i]]) for i in HG.nodes('protein')])
        if flag_monoeffect:
            HG.nodes['monoeffect'].data['feature'] = torch.stack([torch.tensor(entity2embed[monoeffect[i]]) for i in HG.nodes('monoeffect')])
        # if flag_monoeffect_class:
        #     HG.nodes['disease_class'].data['feature'] = torch.stack([torch.tensor(entity2embed[i]) for i in eff_class])
    else:
        feature_dim = 100
        HG.nodes['drug'].data['feature'] = torch.randn(HG.num_nodes('drug'), feature_dim)
        HG.nodes['protein'].data['feature'] = torch.randn(HG.num_nodes('protein'), feature_dim)
        if flag_monoeffect:
            HG.nodes['monoeffect'].data['feature'] = torch.randn(HG.num_nodes('monoeffect'), feature_dim)
        # if flag_monoeffect_class:
        #     HG.nodes['disease_class'].data['feature'] = torch.randn(len(eff_class), feature_dim)

    return HG, drugs, drugs_id, edge_types_id, drug_drug, entity2embed


if __name__ == '__main__':
    read_data()