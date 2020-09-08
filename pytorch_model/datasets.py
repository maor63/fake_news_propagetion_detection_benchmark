from operator import itemgetter

from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class Twitter15Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dir_path, ):
        self.name = 'twitter15'
        self.dir_path = Path(dir_path)
        source_data = pd.read_csv(self.dir_path / f'source_tweets.txt', header=None, sep='\t', names=['tid', 'text'])
        self.source_text_dict = dict(source_data[['tid', 'text']].itertuples(index=False))
        label_data = pd.read_csv(self.dir_path / f'label.txt', header=None, sep=':', names=['label', 'tid'])
        self.label_dict = dict(label_data[['tid', 'label']].itertuples(index=False))
        self.source_tweets = label_data['tid'].values
        self.source_edges = {tid: self.read_graph(tid) for tid in tqdm(self.source_tweets, desc='read graphs')}

    def __len__(self):
        return len(self.source_tweets)

    def __getitem__(self, idx):
        tid = self.source_tweets[idx]
        label = self.label_dict[tid]
        text = self.source_text_dict[tid]
        edges = self.source_edges[tid]
        return tid, text, edges, label

    def read_graph(self, tid):
        col_names = ['left_node', 'right_node']
        propagation_df = pd.read_csv(self.dir_path / 'tree' / f'{tid}.txt', delimiter='->', header=None,
                                     names=col_names, engine="python")
        source_authors = propagation_df['left_node'].apply(eval).apply(itemgetter(0))
        target_authors = propagation_df['right_node'].apply(eval).apply(itemgetter(0))

        edges = list(zip(source_authors, target_authors))

        return edges
t = Twitter15Dataset('datasets/twitter15')
for row in t:
    pass