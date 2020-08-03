import twitter
import pandas as pd
import os
import json
import tqdm

api = twitter.api.Api(consumer_key='consumer_key',
                      consumer_secret='consumer_secret',
                      access_token_key='access_token_key',
                      access_token_secret='access_token_secret',
                      sleep_on_rate_limit=True)

output_path = 'Data/Twitter/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

claims_tweets_df = pd.read_csv('Data/Twitter.txt', sep='\t', header=None, names=['claim', 'label', 'tweets'])
claims_tweets_df['tweets'] = claims_tweets_df['tweets'].str.split()
claims_tweets_df['claim'] = claims_tweets_df['claim'].str.split(':')
claims_tweets_df['label'] = claims_tweets_df['label'].str.split(':')
labels = []
claims = []

claims_tweets_df = claims_tweets_df.iloc[47:]
for i, row in tqdm.tqdm(claims_tweets_df.iterrows(), total=len(claims_tweets_df)):
    claim_id = row['claim'][1]
    label = row['label'][1]
    labels.append(label)
    claims.append(claim_id)
    tweet_ids = row['tweets']

    jsons = []
    tweets = api.GetStatuses(tweet_ids)
    for tweet in tweets:
        jsons.append(tweet._json)
    with open(os.path.join(output_path, f'{claim_id}.json'), 'w') as outfile:
        json.dump(jsons, outfile)


claims_tweets_df = pd.read_csv('Data/Twitter.txt', sep='\t', header=None, names=['claim', 'label', 'tweets'])
claims_tweets_df['tweets'] = claims_tweets_df['tweets'].str.split()
claims_tweets_df['claim'] = claims_tweets_df['claim'].str.split(':')
claims_tweets_df['label'] = claims_tweets_df['label'].str.split(':')
label_df = pd.DataFrame()
label_df['claim'] = claims
label_df['label'] = labels
label_df.to_csv('Data/twitter_labels.csv', index=False)