from datareader import read_data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch
import argparse
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import sys
from model import *
from sklearn import metrics
import json
from tqdm import tqdm, trange


def setseed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class fewshot_train_dataset(Dataset):
    def __init__(self, few, batch, drugs, drugs_id, rel, drug_drug):
        super().__init__()
        self.few = few
        self.rel = rel
        self.batch = batch
        self.drug_drug = drug_drug
        self.n_drug = len(drugs_id)
        self.drugs_id = drugs_id
        self.drugs = drugs

    def __len__(self):
        return len(self.rel)

    def __getitem__(self, item):
        rel = self.rel[item]
        pos = self.drug_drug[rel]
        np.random.shuffle(pos)
        pos_set = set()
        for (i, j) in pos:
            pos_set.add((i, j))
            pos_set.add((j, i))
        drug_pairs = [(i, j) for i in range(self.n_drug) for j in range(self.n_drug) if i < j and (i, j) not in pos_set]
        neg = np.random.choice(list(range(len(drug_pairs))), size=self.batch, replace=True) # negative
        neg = [drug_pairs[i] for i in neg]
        neg = [(self.drugs[i], self.drugs[j]) for (i, j) in neg]
        sup = [(self.drugs[i], self.drugs[j]) for (i, j) in pos[:self.few]] # support
        pos = [(self.drugs[i], self.drugs[j]) for (i, j) in pos[self.few:self.few + self.batch]] # positive
        while len(pos) < self.batch:
            pos = pos + pos
        pos = pos[:self.batch]
        return sup, pos, neg


class fewshot_test_dataset(Dataset):
    def __init__(self, few, drugs, drugs_id, rel, drug_drug):
        super().__init__()
        self.few = few
        self.rel = rel
        self.drug_drug = drug_drug
        self.n_drug = len(drugs_id)
        self.drugs_id = drugs_id
        self.drugs = drugs

    def __len__(self):
        return len(self.rel)

    def __getitem__(self, item):
        rel = self.rel[item]
        pos = self.drug_drug[rel]
        np.random.shuffle(pos)
        pos_set = set()
        for (i, j) in pos:
            pos_set.add((i, j))
            pos_set.add((j, i))
        drug_pairs = [(i, j) for i in range(self.n_drug) for j in range(self.n_drug) if i < j and (i, j) not in pos_set]
        neg = [[self.drugs[i], self.drugs[j]] for (i, j) in drug_pairs] # negative
        sup = [[self.drugs[i], self.drugs[j]] for (i, j) in pos[:self.few]] # support
        pos = [[self.drugs[i], self.drugs[j]] for (i, j) in pos[self.few:]] # positive
        return sup, pos, neg


def _get(data):
    dataset, i = data
    return dataset[i]


def batch_loader(dataset, batch_size=1, num_worker=1, shuffle=False):
    tmp = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(tmp)
    with Pool(num_worker) as P:
        for i in range(0, len(tmp), batch_size):
            s, p, n = [], [], []
            for sup, pos, neg in P.map(_get, [(dataset, j) for j in tmp[i:i+batch_size]]):
                s.append(sup)
                p.append(pos)
                n.append(neg)
            yield s, p, n


def valid(model, dataset):
    data = {'AUROC': 0, 'AUPRC': 0, 'AP@50': 0}
    with torch.no_grad():
        model.eval()
        cnt = 0
        t = tqdm(batch_loader(dataset, num_worker=1), total=len(dataset))
        for sup, pos, neg in t:
            p_score, n_score = model.train_forward(sup, pos, neg)
            p_score = p_score.detach().cpu()
            n_score = n_score.detach().cpu()
            score = torch.cat((p_score.squeeze(0), n_score.squeeze(0))).numpy()
            label = [1] * p_score.shape[1] + [0] * n_score.shape[1]
            rk = np.argsort(score)
            ap = 0
            hit = 0
            for i in range(50):
                if label[rk[i]] == 1:
                    hit += 1
                    ap += hit / (i + 1)
            ap /= min(50, p_score.shape[1])
            data['AP@50'] += ap
            fpr, tpr, thresholds = metrics.roc_curve(label, 1.0 - score)
            data['AUROC'] += metrics.auc(fpr, tpr)
            precision, recall, thresholds = metrics.precision_recall_curve(label, 1.0 - score)
            data['AUPRC'] += metrics.auc(recall, precision)
            cnt += 1
            info = "AUROC: {:.3f}\tAUPRC: {:.3f}\tAP@50: {:.3f}".format(data['AUROC'] / cnt, data['AUPRC'] / cnt, data['AP@50'] / cnt)
            t.set_description(info)
            t.refresh()
    for k in data:
        data[k] /= cnt
    return data


if __name__ == '__main__':
    setseed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--few', required=False, default=5, type=int)
    parser.add_argument('--num_query', required=False, default=10, type=int)
    parser.add_argument('--batch', required=False, default=20, type=int)
    parser.add_argument('--epoch', required=False, default=1000, type=int)
    parser.add_argument('--patience', required=False, default=30, type=int)
    parser.add_argument('--data', required=False, default='data/', type=str)
    parser.add_argument('--device', required=False, default='cpu', type=str)
    parser.add_argument('--log_step', required=False, default=30, type=int)
    parser.add_argument('--dropout', required=False, default=0.5, type=float)
    parser.add_argument('--margin', required=False, default=2.0, type=float)
    parser.add_argument('--metric', required=False, default='AUPRC', type=str)
    parser.add_argument('--task', required=False, default='test', type=str)
    args = parser.parse_args()

    HG, drugs, drugs_id, edge_types_id, drug_drug, pretrain_embed = read_data(args.data)
    all_relation = sorted(list(edge_types_id.values()))
    _, test_rel = train_test_split(all_relation, test_size=0.1, random_state=0)
    train_rel, valid_rel = train_test_split(_, test_size=0.11111, random_state=0)

    train_dataset = fewshot_train_dataset(args.few, args.num_query, drugs, drugs_id, train_rel, drug_drug)
    valid_dataset = fewshot_test_dataset(args.few, drugs, drugs_id, valid_rel, drug_drug)
    test_dataset = fewshot_test_dataset(args.few, drugs, drugs_id, test_rel, drug_drug)

    if args.task == 'train':
        train_loader = batch_loader(train_dataset, batch_size=args.batch, shuffle=True, num_worker=1)

        model = Model(pretrain_embed, args.few, in_features=100, dropout=args.dropout, device=args.device)
        model.to(args.device)
        model.train()
        loss_fn = nn.MarginRankingLoss(margin=args.margin)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-3)
        best_score = None
        best_epoch = 0

        for ep in range(args.epoch):
            model.train()
            for sup, pos, neg in train_loader:
                optimizer.zero_grad()
                p_score, n_score = model.train_forward(sup, pos, neg)
                loss = loss_fn(p_score.view(-1), n_score.view(-1), -torch.ones_like(p_score).view(-1))
                loss.backward()
                optimizer.step()
                break
            if ep % 10 == 0:
                print('Epoch', ep, f'loss={loss}')
            if ep > 0 and args.log_step != 0 and ep % args.log_step == 0:
                print('=' * 10, 'Validation', '=' * 10)
                res = valid(model, valid_dataset)
                if best_score is None or best_score < res[args.metric]:
                    best_epoch = ep
                    best_score = res[args.metric]
                    torch.save(model, 'best_model.pt')
                    json.dump(res, open('validation_results.json', 'w'))
                print(f'Current best {args.metric}={best_score}, updated {ep - best_epoch} epochs ago')
                if ep - best_epoch >= args.patience * args.log_step:
                    break

    print('=' * 10, 'Testing', '=' * 10)
    model = torch.load('best_model.pt')
    model.to(args.device)
    res = valid(model, test_dataset)
    json.dump(res, open('testing_results.json', 'w'))
