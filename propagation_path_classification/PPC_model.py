import sys, os, json, time, datetime
import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold

sys.path.append('..')
# import utils

# project_folder = os.path.join('..', '..')

import numpy as np
import random
import tensorflow as tf
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Concatenate, TimeDistributed, LSTM, \
    AveragePooling1D, Embedding, GRU, GlobalAveragePooling1D
from keras import initializers, regularizers
from keras.initializers import RandomNormal
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn import preprocessing

conv_len = 5
cut_off = 0.5


def roc_auc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)


def normalize(x, positions):
    num_columns = x.shape[1]
    for i in range(num_columns):
        if i in positions:
            x[:, i:i + 1] = np.copy(preprocessing.robust_scale(x[:, i:i + 1]))
    return x


def rc_model(input_shape):
    input_f = Input(shape=(input_shape[0], input_shape[1],), dtype='float32', name='input_f')
    rnn = GRU(32, return_sequences=True)(input_f)
    rnn = GlobalAveragePooling1D()(rnn)

    cnn = Conv1D(32, conv_len, activation='relu')(input_f)
    cnn = MaxPooling1D(3)(cnn)
    cnn = GlobalAveragePooling1D()(cnn)

    rnn_cnn = Concatenate()([rnn, cnn])

    dense = Dense(32, activation='relu')(rnn_cnn)
    dense = Dropout(0.25)(dense)
    output_f = Dense(1, activation='sigmoid', name='output_f')(dense)
    model = Model(inputs=[input_f], outputs=[output_f])
    return model


def model_train(model, file_name, data, epochs, batch_size):
    model.compile(loss={'output_f': 'binary_crossentropy'}, optimizer='adam', metrics=['accuracy'])
    call_back = ModelCheckpoint(file_name, monitor='val_acc', verbose=0, save_best_only=True,
                                save_weights_only=False,
                                mode='auto', period=1)
    input_train = {'input_f': data['x_train']}
    output_train = {'output_f': data['y_train']}
    # input_valid = {'input_f': data['x_valid']}
    # output_valid = {'output_f': data['y_valid']}
    model.fit(input_train, output_train, epochs=epochs, batch_size=batch_size, validation_split=0.15,
              callbacks=[call_back], verbose=2)
    return model


def model_predict(model, x):
    input_test = {'input_f': x}
    pred_test = model.predict(input_test)
    pred_test = pred_test.reshape((pred_test.shape[0],))
    return pred_test


def compute_metrics(pred_test, y_test):
    auc = round(roc_auc_score(y_test, pred_test), 4)
    print(f'AUC: {auc}')
    y_pred = np.copy(pred_test)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    acc = round(accuracy_score(y_test, y_pred.ravel().astype(int)), 4)
    f1 = round(f1_score(y_test, y_pred.ravel().astype(int)), 4)
    prec = round(precision_score(y_test, y_pred.ravel().astype(int)), 4)
    recall = round(recall_score(y_test, y_pred.ravel().astype(int)), 4)
    print(f'Accuracy: {acc}')
    print(f'F1: {f1}')
    print(f'Precision: {prec}')
    print(f'Recall: {recall}')
    return [auc, acc, f1, prec, recall]
    # tp_1, tn_1, fp_1, fn_1, tp_0, tn_0, fp_0, fn_0 = 0, 0, 0, 0, 0, 0, 0, 0
    #
    # for i in range(pred_test.shape[0]):
    #     lp = pred_test[i]
    #     lt = y_test[i]
    #     if lp >= cut_off:
    #         lp = 1
    #     else:
    #         lp = 0
    #     if lp == 1 and lt == 1:
    #         tp_1 += 1
    #         tn_0 += 1
    #     if lp == 0 and lt == 0:
    #         tn_1 += 1
    #         tp_0 += 1
    #     if lp == 1 and lt == 0:
    #         fp_1 += 1
    #         fn_0 += 1
    #     if lp == 0 and lt == 1:
    #         fn_1 += 1
    #         fp_0 += 1
    #
    # acc = (tp_1 + tn_1) / (tp_1 + tn_1 + fp_1 + fn_1)
    # acc_0 = (tp_0 + tn_0) / (tp_0 + tn_0 + fp_0 + fn_0)
    # if acc != acc_0:
    #     print('error')
    #
    # try:
    #     pre_1 = tp_1 / (tp_1 + fp_1)
    #     rec_1 = tp_1 / (tp_1 + fn_1)
    #     f_1 = 2 * tp_1 / (2 * tp_1 + fp_1 + fn_1)
    #
    #     pre_0 = tp_0 / (tp_0 + fp_0)
    #     rec_0 = tp_0 / (tp_0 + fn_0)
    #     f_0 = 2 * tp_0 / (2 * tp_0 + fp_0 + fn_0)
    # except:
    #     return None
    #
    # # res = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(acc, pre_1, rec_1, f_1, pre_0, rec_0, f_0)
    # res = [acc, pre_1, rec_1, f_1, pre_0, rec_0, f_0]
    #
    # return res


def model_evaluate_for_claims(model, x, y, tweet_ids):
    input_test = {'input_f': x}
    pred_test = model.predict(input_test)
    y_test = y
    claim_to_post_df = pd.read_csv('datasets/fake_news_1000_retweet_path_by_date/claim_to_post_dict.csv')
    claim_to_post_df = claim_to_post_df.set_index('post_id').loc[tweet_ids].reset_index()

    tweet_to_label = pd.Series(index=tweet_ids, data=y)
    tweet_to_pred = pd.Series(index=tweet_ids, data=pred_test.reshape(1, -1).squeeze())
    claim_to_post_df['label'] = claim_to_post_df['post_id'].apply(lambda x: tweet_to_label[x])
    claim_to_post_df['pred'] = claim_to_post_df['post_id'].apply(lambda x: tweet_to_pred[x])

    # claim_to_post_df['label'] = y
    # claim_to_post_df['pred'] = pred_test
    labels = claim_to_post_df.groupby('claim_id')['label'].agg(pd.Series.mode)
    preds = claim_to_post_df.groupby('claim_id')['pred'].mean()
    rs = compute_metrics(preds, labels)
    return rs


def model_evaluate(model, x, y):
    input_test = {'input_f': x}
    pred_test = model.predict(input_test)
    y_test = y
    rs = compute_metrics(pred_test, y_test)
    return rs


def new_main():
    # seq_len = 35
    # epochs = 100
    to_train_model = False
    epochs = 100
    batch_size = 128
    nb_sample = 1
    seq_lens = [100]
    data_opt = 'twitter15'
    data_opt = 'fake_news_1000_retweet_path_by_date'
    # data_opt = 'fake_news_17k_prop_data'

    if data_opt == 'twitter':
        data_name = 'twitter15'
    else:
        data_name = 'weibo'

    X = np.load(f'processed_datasets/{data_opt}/X.npy')
    y = np.load(f'processed_datasets/{data_opt}/y.npy')
    tweet_ids = np.load(f'processed_datasets/{data_opt}/tweet_ids.npy')
    # print(x.shape, y.shape)

    n = X.shape[0]
    nb_feature = X.shape[2]
    x = X.astype('float32')
    pos = np.arange(n)

    rs_avg = []
    claims_results = []

    # for sample in range(nb_sample):
    skf = StratifiedKFold(n_splits=10)

    shape = x.shape
    x = x.reshape([shape[0] * shape[1], shape[2]])

    if 'twitter' in data_opt or 'fake_news' in data_opt:
        pos_norm = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    else:
        pos_norm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    x = normalize(x, pos_norm)
    x = x.reshape([shape[0], shape[1], shape[2]])

    for sample, (train_index, test_index) in enumerate(skf.split(x, y)):
        print('sample {}'.format(sample))


        for seq_len in seq_lens:
            data = {}

            data['x_train'] = x[train_index, 0:seq_len, :]
            data['y_train'] = y[train_index]

            data['x_test'] = x[test_index, 0:seq_len, :]
            data['y_test'] = y[test_index]

            model = rc_model(input_shape=[seq_len, nb_feature])
            # model.summary()

            model_folder = os.path.join('processed_datasets/', data_opt)
            model_name = 'sp_{}_seqlen_{}'.format(sample, seq_len)
            if to_train_model:
                model_train(model, file_name=os.path.join(model_folder, model_name), data=data, epochs=epochs,
                            batch_size=batch_size)
            best_model = load_model(os.path.join(model_folder, model_name))
            # print('#########Validation###########')
            # model_evaluate(best_model, x=data['x_test'], y=data['y_test'])
            print('#########Test###########')
            rs = model_evaluate(best_model, x=data['x_test'], y=data['y_test'])
            rs_avg.append(rs)

            claim_results = model_evaluate_for_claims(best_model, data['x_test'], data['y_test'], tweet_ids[test_index])
            claims_results.append(claim_results)

    avg = np.array(rs_avg).mean(axis=0)
    print('AUC\t ACC\t F1\t Precision\t Recall')
    print('\t'.join(map(str, np.around(avg, 4))))
    print('AUC\t ACC\t F1\t Precision\t Recall')
    print('\t'.join(map(str, np.around(np.array(claims_results).mean(axis=0), 4))))


if __name__ == '__main__':
    new_main()
