import os
from operator import itemgetter
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import networkx as nx
from karateclub import FeatherGraph, Graph2Vec, FGSD, GL2Vec, SF, NetLSD, GeoScattering
from keras import *
from keras.layers import *
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def get_retweet_graph(tweet_propagation_path, propagation_size, tree_delimiter='->', time_limit=None):
    col_names = ['left_node', 'right_node']
    propagation_df = pd.read_csv(tweet_propagation_path, delimiter=tree_delimiter, header=None, names=col_names)
    source_authors = propagation_df['left_node'].apply(eval).apply(itemgetter(0))
    target_authors = propagation_df['right_node'].apply(eval).apply(itemgetter(0))

    node_to_idx = LabelEncoder().fit(list(set(source_authors) | set(target_authors)))

    source_authors = node_to_idx.transform(source_authors)
    target_authors = node_to_idx.transform(target_authors)
    edges = list(zip(source_authors, target_authors))
    if len(edges) < len(target_authors):
        pass

    root_author = target_authors[0]
    root_egdes = [(root_author, src) for src in source_authors[1:]]

    retweet_graph = nx.DiGraph()
    retweet_graph.add_edges_from(list(set(edges[1:] + root_egdes)))
    return retweet_graph


def get_label_dict(label_file, title='', posts_df=None):
    labels_df = pd.read_csv(label_file, sep=':', header=None, names=['label', 'tweet_id'])
    # labels_df.groupby('label')['tweet_id'].count().rename({'tweet_id': 'count'}).plot.pie(title=title)
    # plt.show()
    print(labels_df.groupby('label')['tweet_id'].count())
    label_dict = dict(zip(labels_df['tweet_id'], labels_df['label']))
    return label_dict


def str_to_timestamp(string):
    # structured_time = time.strptime(string, "%a %b %d %H:%M:%S +0000 %Y")
    structured_time = datetime.strptime(string, "%a %b %d %H:%M:%S +0000 %Y")
    epoch = datetime(1970, 1, 1)
    diff = structured_time - epoch
    timestamp = diff.total_seconds()
    return timestamp


def extract_user_features(users_df, user_id_field='author_osn_id'):
    users_features_df = pd.DataFrame(index=users_df[user_id_field])
    users_features_df.index = users_features_df.index.astype('str')
    users_features_df['name_size'] = users_df['name'].str.len().values
    users_features_df['screen_name_size'] = users_df['author_screen_name'].str.len().values
    users_features_df['description_size'] = users_df['description'].str.len().values
    users_features_df['followers_count'] = users_df['followers_count'].values
    users_features_df['friends_count'] = users_df['friends_count'].values
    users_features_df['listed_count'] = users_df['listed_count'].values
    users_features_df['favourites_count'] = users_df['favourites_count'].values
    users_features_df['statuses_count'] = users_df['statuses_count'].values

    time_now = 'Thu Jan 11 00:00:00 +0000 2018'
    # time_now = datetime.strptime('2019-05-29 00:00:00', '%Y-%m-%d %H:%M:%S').strftime("%a %b %d %H:%M:%S +0000 %Y")
    # time_now = 'Thu Jan 29 00:00:00 +0000 2020'

    ts_now = str_to_timestamp(time_now)
    ts_user = users_df['created_at'].apply(str_to_timestamp)
    users_features_df['account_age'] = ((ts_now - ts_user) / (3600 * 24)).values

    users_features_df['has_url'] = users_df['url'].notnull().astype(np.int8).values
    users_features_df['protected'] = users_df['protected'].astype(np.int8).values
    users_features_df['geo_enabled'] = users_df['geo_enabled'].astype(np.int8).values
    users_features_df['verified'] = users_df['verified'].astype(np.int8).values
    users_features_df['profile_use_background_image'] = users_df['profile_background_tile'].astype(np.int8).values
    # f.append(int(user['profile_use_background_image']))
    # users_features_df['has_extended_profile'] = users_df['has_extended_profile'].astype(np.int8)
    # f.append(int(user['has_extended_profile']))
    users_features_df['default_profile'] = users_df['default_profile'].fillna(0).astype(np.int8).values
    users_features_df['default_profile_image'] = users_df['default_profile_image'].fillna(0).astype(np.int8).values
    users_features_df = users_features_df.fillna(0)
    return users_features_df


def xgb_acc(predt: np.ndarray, dtrain: xgb.DMatrix):
    ''' Root mean squared log error metric.'''
    y = dtrain.get_label()
    return 'Acc', accuracy_score(y, predt.argmax(axis=0))


def convert_trees_to_graphs(data_path, dataset_sufix, path_len, possible_labels, time_limit, tree_delimiter,
                            user_id_field):
    # users_df =
    users_features_dfs = []
    count = 0
    authors_file_name = list(filter(lambda name: 'authors' in name, os.listdir(data_path)))[0]
    for users_df in pd.read_csv(data_path / authors_file_name, chunksize=10000):
        print(f'\r read authors {count}', end='')
        count += users_df.shape[0]
        users_features_dfs.append(extract_user_features(users_df, user_id_field))
    print()
    users_features_df = pd.concat(users_features_dfs)
    users_features_df = users_features_df.loc[~users_features_df.index.duplicated(keep='first')]
    # posts_file_name = list(filter(lambda name: 'posts' in name, os.listdir(data_path)))[0]
    # posts_df = pd.read_csv(data_path / posts_file_name)
    label_dict = get_label_dict(data_path / 'label.txt', title=dataset_sufix)
    y = []
    graphs = []
    tweet_ids = []
    all_users_ids = set(users_features_df.index)
    total_tweets = len(label_dict)
    for i, tweet_id in enumerate(label_dict):
        print(f'\rprocess tweets features {i + 1} / {total_tweets}', end='')
        tweet_propagation_path = data_path / 'tree' / f'{tweet_id}.txt'
        if os.path.exists(tweet_propagation_path):
            if label_dict[tweet_id] in possible_labels:
                y.append(label_dict[tweet_id])
            else:
                continue
            retweet_graph = get_retweet_graph(tweet_propagation_path, path_len, tree_delimiter, time_limit)
            graphs.append(retweet_graph)
    print()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print(f'Label encoding: {dict(zip(possible_labels, label_encoder.transform(possible_labels)))}')
    return graphs, label_encoder, y


def build_model(input_size, output_size):
    input_f = Input(shape=(input_size, ), dtype='float32', name='input_f')
    rc = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.000001))(input_f)
    # rc = Dropout(0.5)(rc)
    output_f = Dense(output_size, activation='softmax', name='output_f')(rc)
    model = Model(inputs=[input_f], outputs=[output_f])
    return model


def main():
    # dataset_sufix = 'fake_news_17k_prop_data'
    # dataset_sufix = 'fake_news_1000_retweet_path_by_date'
    dataset_sufix = 'twitter16'
    # dataset_sufix = 'twitter15'
    path_len = 100
    time_limit = 24 * 60  # None for all
    possible_labels = ['unverified', 'non-rumor', 'true', 'false']
    # possible_labels = [False, True]
    # possible_labels = ['non-rumor', 'false']
    # false_labels = {'false'}
    # false_labels = {False}
    # true_labels = {'non-rumor'}
    # true_labels = {True}
    # tree_delimiter = '-'
    tree_delimiter = '->'
    # user_id_field = 'author_guid'
    user_id_field = 'author_osn_id'

    output_features_path = Path(os.path.join('processed_datasets/', dataset_sufix))
    data_path = Path(os.path.join('datasets/', dataset_sufix))

    graphs, label_encoder, y = convert_trees_to_graphs(data_path, dataset_sufix, path_len, possible_labels, time_limit,
                                                       tree_delimiter, user_id_field)

    undirected_graphs = [g.to_undirected() for g in graphs]

    for graph_emb in [
        # FeatherGraph,
        Graph2Vec,
        # FGSD,
        # # GL2Vec,
        # SF,
        # NetLSD,
        # GeoScattering
    ]:
        model = graph_emb()
        model.fit(undirected_graphs)
        X = model.get_embedding()

        f1s = []

        macro_precision = []
        macro_recall = []
        accs = []
        skf = StratifiedKFold(5)
        epochs = 10
        batch_size = 10
        for train_index, test_index in skf.split(X, y):
            model = build_model(X.shape[1], len(np.unique(y)))
            model.compile(loss={'output_f': 'categorical_crossentropy'}, optimizer='adam', metrics=['accuracy'])
            # bst = xgb.XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=100)
            model.fit(X[train_index], to_categorical(y[train_index]), epochs=epochs, batch_size=batch_size,
              validation_split=0.1, verbose=2)
            y_pred = model.predict(X[test_index])
            report_dict = classification_report(y[test_index], y_pred.argmax(axis=1), output_dict=True)
            print(report_dict)

            f1s.append([report_dict[str(pos_label)]['f1-score'] for pos_label in range(len(label_encoder.classes_))])

            macro_precision.append(report_dict['macro avg']['precision'])
            macro_recall.append(report_dict['macro avg']['recall'])
            accs.append(report_dict['accuracy'])

        f1s = np.array(f1s).mean(axis=0)
        print(' \t '.join(['dataset', 'model', 'ACC', 'precision', 'recall'] + [f'f1_{i}' for i in
                                                                                range(len(label_encoder.classes_))]))
        scores = [dataset_sufix, model.__class__.__name__, np.mean(accs), np.mean(macro_precision),
                  np.mean(macro_recall), *f1s]
        print(' \t '.join(map(str, scores)))


if __name__ == '__main__':
    main()
