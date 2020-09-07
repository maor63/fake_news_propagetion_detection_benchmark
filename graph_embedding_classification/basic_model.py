import dgl

from graph_embedding_classification.twitter16_dataset import Twitter16
from sklearn.preprocessing import LabelEncoder
dataset = Twitter16(raw_dir='graph_embedding_classification/')


dgl_graphs = []
for graph, label in dataset:
    u, v = [], []
    for node, node_data in graph.items():
        if node_data['parent'] != 'None':
            u.append(node_data['parent'])
            v.append(str(node))
    lb = LabelEncoder().fit(set(u + v))
    u = lb.transform(u), lb.transform(v)

    g = dgl.graph((u, v))
    pass
