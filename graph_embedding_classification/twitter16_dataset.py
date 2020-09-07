from random import random

import dgl
from dgl.data import DGLDataset
import os
from pathlib import Path
from dgl.data.utils import download
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch
from dgl.nn.pytorch import GraphConv, GINConv, GatedGraphConv, GATConv, TAGConv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Twitter16(DGLDataset):
    def __init__(self,
                 # url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(Twitter16, self).__init__(name='twitter16_TD',
                                        # url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def download(self):
        # download raw data to local disk
        graph_url = 'https://raw.githubusercontent.com/TianBian95/BiGCN/master/data/Twitter16/data.TD_RvNN.vol_5000.txt'
        label_url = 'https://raw.githubusercontent.com/TianBian95/BiGCN/master/data/Twitter16/Twitter16_label_All.txt'
        file_path = Path(self.raw_dir)
        if not file_path.exists():
            os.makedirs(file_path)

        if not os.path.exists(file_path / (self.name + '.graph')):
            download(graph_url, path=file_path / (self.name + '.graph'))
        if not os.path.exists(file_path / (self.name + '.label')):
            download(label_url, path=file_path / (self.name + '.label'))
        pass

    def _load_graph(self, mat_path):
        # print("reading tree"),  ## X
        treeDic = {}
        for line in tqdm(open(mat_path), desc='loading graph'):
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
            if not eid in treeDic:
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
        print('tree no:', len(treeDic))
        return treeDic

    def _load_labels(self, labels_path):
        # print("loading tree label", )
        labelDic = {}
        for line in tqdm(open(labels_path), desc='load graph label:'):
            line = line.rstrip()
            label, eid = line.split('\t')[0], line.split('\t')[2]
            labelDic[eid] = label.lower()
        print(len(labelDic))
        return labelDic

    def process(self):
        self.graphs = self._load_graph(self.raw_path + '.graph')
        self.label = self._load_labels(self.raw_path + '.label')
        self.eids = list(self.label.keys())

    def __getitem__(self, idx):
        # get one example by index
        eid = self.eids[idx]
        return self.graphs[eid], self.label[eid]

    def __len__(self):
        # number of data examples
        return len(self.graphs)

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass


dataset = Twitter16(raw_dir='graph_embedding_classification/')


dgl_graphs = []
labels = []
for graph, label in dataset:
    u, v = [], []
    for node, node_data in graph.items():
        if node_data['parent'] != 'None':
            u.append(node_data['parent'])
            v.append(str(node))
    lb = LabelEncoder().fit(u + v)
    u, v = lb.transform(u), lb.transform(v)

    if len(u) > 0:
        g = dgl.graph((u, v))
        dgl_graphs.append(g)
        labels.append(label)
    pass

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim,)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g: dgl.DGLGraph):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.out_degrees().view(-1, 1).float()
        # E = torch.zeros(len(g.edges))
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)


n_classes = 4
model = Classifier(1, 64, n_classes)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
model.train()

labels = LabelEncoder().fit_transform(labels)

epoch_losses = []
for epoch in range(80):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(zip(dgl_graphs, labels), 1):
        bg = dgl.add_self_loop(bg)
        prediction = model(bg)
        loss = loss_func(prediction, torch.tensor([label]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter)
    with torch.no_grad():
        predictions = [torch.argmax(model(g)) for g in dgl_graphs]
        train_acc = accuracy_score(labels, predictions)

        # test_predictions = [torch.argmax(model(g)) for g in test_dgl_graphs]
        # test_acc = accuracy_score(y_test, test_predictions)
    print('Epoch {}, loss {:.4f}, train acc: {}, test acc: {}'.format(epoch, epoch_loss, train_acc, 0))
    epoch_losses.append(epoch_loss)

# plt.title('cross entropy averaged over minibatches')
# plt.plot(epoch_losses)
# plt.show()