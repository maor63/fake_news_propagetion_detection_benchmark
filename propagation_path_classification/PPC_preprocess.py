import os
from operator import itemgetter
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


def get_retweet_path(tweet_propagation_path, propagation_size, tree_delimiter='->', time_limit=None):
    col_names = ['left_node', 'right_node']
    propagation_df = pd.read_csv(tweet_propagation_path, delimiter=tree_delimiter, header=None, names=col_names)
    # convert strings to lists
    # user_ids = propagation_df['right_node'].apply(eval).apply(itemgetter(0)).apply(int).tolist()[:propagation_size]
    sort_users_by_propagation = sorted(propagation_df['right_node'].apply(eval), key=lambda x: float(x[2]))
    if time_limit:
        sort_users_by_propagation = [u for u in sort_users_by_propagation if float(u[2]) <= time_limit]
    user_ids = list(map(itemgetter(0), sort_users_by_propagation))[:propagation_size]
    # user_ids = propagation_df['right_node'].apply(eval).apply(itemgetter(0)).tolist()
    return user_ids


def get_label_dict(label_file, title='', posts_df=None):
    labels_df = pd.read_csv(label_file, sep=':', header=None, names=['label', 'tweet_id'])
    # if posts_df is not None:
    #     df = posts_df.merge(labels_df, left_on='post_id', right_on='tweet_id')
    #     res = df.groupby('domain')['verdict'].sum()

    # labels_df.groupby('label')['tweet_id'].count().rename({'tweet_id': 'count'}).plot.pie(title=title, autopct='%')
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


def main():
    # dataset_sufix = 'fake_news_17k_prop_data'
    # dataset_sufix = 'fake_news_1000_retweet_path_by_date'
    dataset_sufix = 'fake_news_1000_retweet_path_by_friend_con'
    # dataset_sufix = 'twitter16'
    path_len = 100
    time_limit = 24 * 60 # None for all
    # possible_labels = ['unverified', 'non-rumor', 'true', 'false']
    possible_labels = [False, True]
    # possible_labels = ['non-rumor', 'false']
    false_labels = {'false'}
    # false_labels = {False}
    true_labels = {'non-rumor'}
    # true_labels = {True}
    tree_delimiter = '-'
    # tree_delimiter = '->'
    user_id_field = 'author_guid'
    # user_id_field = 'author_osn_id'

    output_features_path = Path(os.path.join('processed_datasets/', dataset_sufix))
    data_path = Path(os.path.join('datasets/', dataset_sufix))

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
    X = []
    tweet_ids = []

    all_users_ids = set(users_features_df.index)
    total_tweets = len(label_dict)
    for i, tweet_id in enumerate(label_dict):
        print(f'\rprocess tweets features {i + 1} / {total_tweets}', end='')
        tweet_propagation_path = data_path / 'tree' / f'{tweet_id}.txt'
        if os.path.exists(tweet_propagation_path):
            # if label_dict[tweet_id] in false_labels:
            #     y.append(1)
            # elif label_dict[tweet_id] in true_labels:
            #     y.append(0)
            # else:
            #     continue
            if label_dict[tweet_id] in possible_labels:
                y.append(label_dict[tweet_id])
            else:
                continue
            user_ids_in_propagation = get_retweet_path(tweet_propagation_path, path_len, tree_delimiter, time_limit)
            user_ids_in_propagation = [user_id for user_id in user_ids_in_propagation if user_id in all_users_ids]
            feature_vector = users_features_df.loc[user_ids_in_propagation]
            if len(feature_vector) < path_len:
                missing_features_size = (path_len - len(feature_vector), len(feature_vector.columns))
                dummy_features_df = pd.DataFrame(np.zeros(missing_features_size), columns=feature_vector.columns)
                feature_vector = feature_vector.append(dummy_features_df)

            X.append(feature_vector.values)
            tweet_ids.append(tweet_id)
    print()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print(f'Label encoding: {dict(zip(possible_labels, label_encoder.transform(possible_labels)))}')
    y = np.array(y)
    X = np.array(X)
    if not os.path.exists(output_features_path):
        os.makedirs(output_features_path)
    print(X.shape, y.shape)
    np.save(output_features_path / 'X.npy', X)
    np.save(output_features_path / 'y.npy', y)
    np.save(output_features_path / 'tweet_ids.npy', np.array(tweet_ids))


if __name__ == '__main__':
    main()
