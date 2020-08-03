import os
import pickle
import torch
from sklearn.metrics import classification_report
from GLAN_yang_2019.model.GLAN import GLAN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_dataset(task):
    X_train_tid, X_train, y_train, word_embeddings, adj = pickle.load(
        open("GLAN_yang_2019/dataset/" + task + "/train.pkl", 'rb'))
    X_dev_tid, X_dev, y_dev = pickle.load(open("GLAN_yang_2019/dataset/" + task + "/dev.pkl", 'rb'))
    X_test_tid, X_test, y_test = pickle.load(open("GLAN_yang_2019/dataset/" + task + "/test.pkl", 'rb'))
    config['embedding_weights'] = word_embeddings
    print("#nodes: ", adj.shape[0])
    return X_train_tid, X_train, y_train, \
           X_dev_tid, X_dev, y_dev, \
           X_test_tid, X_test, y_test, adj


def train_and_test(model, task):
    model_suffix = model.__name__.lower().strip("text")
    config['save_path'] = 'GLAN_yang_2019/checkpoint/weights.best.' + task + "." + model_suffix

    X_train_tid, X_train, y_train, \
    X_dev_tid, X_dev, y_dev, \
    X_test_tid, X_test, y_test, adj = load_dataset(task)

    nn = model(config, adj)
    nn.fit(X_train_tid, X_train, y_train,
           X_dev_tid, X_dev, y_dev)

    print("================================")
    nn.load_state_dict(torch.load(config['save_path']))
    y_pred = nn.predict(X_test_tid, X_test)
    print(classification_report(y_test, y_pred, target_names=config['target_names'], digits=3))


config = {
    'reg': 0,
    'batch_size': 128,
    'nb_filters': 100,
    'kernel_sizes': [3, 4, 5],
    'dropout': 0.5,
    'maxlen': 50,
    'epochs': 20,
    'num_classes': 4,
    'target_names': ['NR', 'FR', 'UR', 'TR']
}

if __name__ == '__main__':
    task = 'fake_news_1000_retweet_path_by_friend_con'
    # task = 'twitter15'
    # task = 'twitter16'
    # task = 'weibo'
    print("task: ", task)

    if task == 'weibo':
        config['num_classes'] = 2
        config['target_names'] = ['NR', 'FR']
    elif task == 'fake_news_1000_retweet_path_by_friend_con':
        config['num_classes'] = 2
        config['target_names'] = ['True', 'False']

    model = GLAN
    train_and_test(model, task)
