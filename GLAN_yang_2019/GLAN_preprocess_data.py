import os
from pathlib import Path
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split

from propagation_path_classification.PPC_preprocess import get_label_dict, get_retweet_path


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

    output_features_path = Path(os.path.join('GLAN_yang_2019/dataset/', dataset_sufix))
    if not os.path.exists(output_features_path):
        os.makedirs(output_features_path)

    data_path = Path(os.path.join('datasets/', dataset_sufix))

    label_dict = get_label_dict(data_path / 'label.txt', title=dataset_sufix)
    total_tweets = len(label_dict.keys())

    # Load posts data

    # users_df =
    posts_features_dfs = []
    count = 0
    posts_file_name = list(filter(lambda name: 'posts' in name, os.listdir(data_path)))[0]
    for posts_df in pd.read_csv(data_path / posts_file_name, chunksize=10000):
        print(f'\r read posts {count}', end='')
        count += posts_df.shape[0]
        posts_features_dfs.append(posts_df)
    print()
    posts_features_df = pd.concat(posts_features_dfs)

    source_tweets_ids = set(label_dict.keys())
    source_tweets_df = posts_features_df[posts_features_df['domain'] != 'retweet'][['post_id', 'content']]
    y = source_tweets_df['post_id'].apply(label_dict.get)
    source_tweets_df['label'] = y

    train_tweets, test_tweets, train_y, test_y = train_test_split(source_tweets_df, y, test_size=0.25, stratify=y)
    train_tweets, dev_tweets, train_y, dev_y = train_test_split(train_tweets, train_y, test_size=0.1, stratify=train_y)

    train_tweets.to_csv(output_features_path / f'{dataset_sufix}.train', index=False, header=False, sep='\t')
    test_tweets.to_csv(output_features_path / f'{dataset_sufix}.test', index=False, header=False, sep='\t')
    dev_tweets.to_csv(output_features_path / f'{dataset_sufix}.dev', index=False, header=False, sep='\t')

    graph_rows = []

    all_user_ids = []
    for i, tweet_id in enumerate(label_dict):
        print(f'\rprocess tweets features {i + 1} / {total_tweets}', end='')
        tweet_propagation_path = data_path / 'tree' / f'{tweet_id}.txt'
        user_ids_in_propagation = get_retweet_path(tweet_propagation_path, path_len, tree_delimiter, time_limit)
        all_user_ids += user_ids_in_propagation

    most_common_users = set([user_id for user_id, _ in Counter(all_user_ids).most_common(40000)])

    for i, tweet_id in enumerate(label_dict):
        print(f'\rprocess tweets features {i + 1} / {total_tweets}', end='')
        tweet_propagation_path = data_path / 'tree' / f'{tweet_id}.txt'
        user_ids_in_propagation = get_retweet_path(tweet_propagation_path, path_len, tree_delimiter, time_limit)
        user_count = Counter(user_ids_in_propagation)
        max_user_count = max(user_count.values())

        prop = [f'{user_id}:{user_count[user_id] / float(max_user_count)}' for user_id in user_ids_in_propagation if user_id in most_common_users]
        prop_str = ' '.join(prop)

        graph_rows.append([tweet_id, prop_str])

    graph_df = pd.DataFrame(graph_rows)
    graph_df.to_csv(output_features_path / f'{dataset_sufix}_graph.txt', index=False, header=False, sep='\t')






if __name__ == '__main__':
    main()
