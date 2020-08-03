import os
import random
from pathlib import Path
import dgl
import torch
from dgl.nn.pytorch import GraphConv, GINConv, GatedGraphConv, GATConv, TAGConv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import networkx as nx
import  numpy as np

import graph_embedding_classification


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim,)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g: dgl.DGLGraph):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.in_degrees().view(-1, 1).float()
        # E = torch.zeros(len(g.edges))
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)


def collate(graphs, labels, batch_size):
    batched_graph = []
    batched_labels = []
    for i, (g, l) in enumerate(zip(graphs, labels), 1):
        batched_graph.append(g)
        batched_labels.append(l)
        if i % batch_size == 0:
            yield batched_graph, torch.tensor(batched_labels)
            batched_graph = []
            batched_labels = []
    yield batched_graph, torch.tensor(batched_labels)


def main():
    # dataset_sufix = 'fake_news_17k_prop_data'
    # dataset_sufix = 'fake_news_1000_retweet_path_by_date'
    dataset_sufix = 'fake_news_1000_retweet_path_by_friend_con'
    # dataset_sufix = 'twitter16'
    # dataset_sufix = 'twitter15'
    path_len = 100
    time_limit = 24 * 60  # None for all
    # possible_labels = ['unverified', 'non-rumor', 'true', 'false']
    possible_labels = [False, True]
    # possible_labels = ['non-rumor', 'false']
    # false_labels = {'false'}
    # false_labels = {False}
    # true_labels = {'non-rumor'}
    # true_labels = {True}
    tree_delimiter = '-'
    # tree_delimiter = '->'
    user_id_field = 'author_guid'
    # user_id_field = 'author_osn_id'

    output_features_path = Path(os.path.join('processed_datasets/', dataset_sufix))
    data_path = Path(os.path.join('datasets/', dataset_sufix))
    graphs, label_encoder, y = graph_embedding_classification.convert_trees_to_graphs(data_path, dataset_sufix, path_len, possible_labels, time_limit,
                                                       tree_delimiter, user_id_field)

    undirected_graphs = [g.to_undirected() for g in graphs]
    ##############################################

    graphs, y = graphs_to_dgl_graphs(graphs, y)
    train_dgl_graphs, test_dgl_graphs, y_train, y_test = train_test_split(graphs, y, test_size=0.2)

    # train_dgl_graphs = graphs_to_dgl_graphs(graphs_train)
    # test_dgl_graphs = graphs_to_dgl_graphs(graphs_test)

    n_classes = len(label_encoder.classes_)
    model = Classifier(1, 64, n_classes)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()

    pos = np.arange(len(train_dgl_graphs))

    epoch_losses = []
    for epoch in range(80):
        epoch_loss = 0
        random.shuffle(pos)
        for iter, (bg, label) in enumerate(zip(train_dgl_graphs[pos], y_train[pos]), 1):
            prediction = model(bg)
            loss = loss_func(prediction, torch.tensor([label]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter)
        with torch.no_grad():
            predictions = [torch.argmax(model(g)) for g in train_dgl_graphs]
            train_acc = accuracy_score(y_train, predictions)

            test_predictions = [torch.argmax(model(g)) for g in test_dgl_graphs]
            test_acc = accuracy_score(y_test, test_predictions)
        print('Epoch {}, loss {:.4f}, train acc: {}, test acc: {}'.format(epoch, epoch_loss, train_acc, test_acc))
        epoch_losses.append(epoch_loss)

    plt.title('cross entropy averaged over minibatches')
    plt.plot(epoch_losses)
    plt.show()

    model.eval()
    # Convert a list of tuples to two lists
    test_X, test_Y = map(list, zip(*testset))
    test_bg = dgl.batch(test_X)
    test_Y = torch.tensor(test_Y).float().view(-1, 1)
    probs_Y = torch.softmax(model(test_bg), 1)
    sampled_Y = torch.multinomial(probs_Y, 1)
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
    print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
        (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
    print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
        (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))


def graphs_to_dgl_graphs(graphs, y):
    dgl_graphs = []
    new_y = []
    for g, label in zip(graphs, y):
        assert isinstance(g, nx.DiGraph)
        if len(list(g.edges())) > 0:
            dgl_g = dgl.DGLGraph()
            nodes = list(g.nodes())
            dgl_g.add_nodes(len(nodes))

            u, v = list(zip(*g.edges()))
            dgl_g.add_edges(u, v)
            dgl_graphs.append(dgl_g)
            new_y.append(label)
    return np.array(dgl_graphs), np.array(new_y)


if __name__ == '__main__':
    main()
