import sys, os, json, time, datetime

from sklearn.metrics import *

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
    # dense = Dropout(0.5)(dense)
    output_f = Dense(1, activation='sigmoid', name='output_f')(dense)
    model = Model(inputs=[input_f], outputs=[output_f])
    return model


def model_train(model, file_name, data, epochs, batch_size):
    model.compile(loss={'output_f': 'binary_crossentropy'}, optimizer='rmsprop', metrics=['accuracy'])
    call_back = ModelCheckpoint(file_name, monitor='val_accuracy', verbose=0, save_best_only=True,
                                save_weights_only=False,
                                mode='auto', period=1)
    input_train = {'input_f': data['x_train']}
    output_train = {'output_f': data['y_train']}
    # input_valid = {'input_f': data['x_valid']}
    # output_valid = {'output_f': data['y_valid']}
    model.fit(input_train, output_train, epochs=epochs, batch_size=batch_size, validation_split=0.25,
              callbacks=[call_back], verbose=2)
    return model


def model_predict(model, x):
    input_test = {'input_f': x}
    pred_test = model.predict(input_test)
    pred_test = pred_test.reshape((pred_test.shape[0],))
    return pred_test


def compute_metrics(pred_test, y_test):
    print(f'AUC: {roc_auc_score(y_test, pred_test)}')
    y_pred = np.copy(pred_test)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    print(f'Accuracy: {accuracy_score(y_test, y_pred.ravel().astype(int))}')
    print(f'F1: {f1_score(y_test, y_pred.ravel().astype(int))}')
    print(f'Precision: {precision_score(y_test, y_pred.ravel().astype(int))}')
    print(f'Recall: {recall_score(y_test, y_pred.ravel().astype(int))}')
    tp_1, tn_1, fp_1, fn_1, tp_0, tn_0, fp_0, fn_0 = 0, 0, 0, 0, 0, 0, 0, 0

    for i in range(pred_test.shape[0]):
        lp = pred_test[i]
        lt = y_test[i]
        if lp >= cut_off:
            lp = 1
        else:
            lp = 0
        if lp == 1 and lt == 1:
            tp_1 += 1
            tn_0 += 1
        if lp == 0 and lt == 0:
            tn_1 += 1
            tp_0 += 1
        if lp == 1 and lt == 0:
            fp_1 += 1
            fn_0 += 1
        if lp == 0 and lt == 1:
            fn_1 += 1
            fp_0 += 1

    acc = (tp_1 + tn_1) / (tp_1 + tn_1 + fp_1 + fn_1)
    acc_0 = (tp_0 + tn_0) / (tp_0 + tn_0 + fp_0 + fn_0)
    if acc != acc_0:
        print('error')

    try:
        pre_1 = tp_1 / (tp_1 + fp_1)
        rec_1 = tp_1 / (tp_1 + fn_1)
        f_1 = 2 * tp_1 / (2 * tp_1 + fp_1 + fn_1)

        pre_0 = tp_0 / (tp_0 + fp_0)
        rec_0 = tp_0 / (tp_0 + fn_0)
        f_0 = 2 * tp_0 / (2 * tp_0 + fp_0 + fn_0)
    except:
        return None

    # res = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(acc, pre_1, rec_1, f_1, pre_0, rec_0, f_0)
    res = [acc, pre_1, rec_1, f_1, pre_0, rec_0, f_0]

    return res


def model_evaluate(model, x, y):
    input_test = {'input_f': x}
    pred_test = model.predict(input_test)
    y_test = y
    rs = compute_metrics(pred_test, y_test)
    return rs


def main():
    # seq_len = 35
    # epochs = 100
    epochs = 100
    batch_size = 128
    nb_sample = 10
    seq_lens = [100]
    data_opt = 'twitter15'
    data_opt = 'fake_news_3500_posts'
    # data_opt = 'fake_news_17k_prop_data'

    if data_opt == 'twitter':
        data_name = 'twitter15'
    else:
        data_name = 'weibo'

    x = np.load(f'processed_datasets/{data_opt}/X.npy')
    y = np.load(f'processed_datasets/{data_opt}/y.npy')
    # print(x.shape, y.shape)

    n = x.shape[0]
    nb_feature = x.shape[2]
    x = x.astype('float32')
    pos = np.arange(n)

    rs_avg = {}
    for seq_len in seq_lens:
        rs_avg[seq_len] = [0 for i in range(7)]

    # split_ratio = [0.6, 0.2, 0.2]
    split_ratio = [0.8, 0.2]

    for sample in range(nb_sample):

        print('sample {}'.format(sample))

        np.random.shuffle(pos)

        for seq_len in seq_lens:

            x1 = x[:, 0:seq_len, :]

            shape = x1.shape
            x1 = x1.reshape([shape[0] * shape[1], shape[2]])

            if 'twitter' in data_opt or 'fake_news' in data_opt:
                pos_norm = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            else:
                pos_norm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            x1 = normalize(x1, pos_norm)
            x1 = x1.reshape([shape[0], shape[1], shape[2]])

            data = {}

            data['x_train'] = x1[pos[0:int(n * split_ratio[0])], :]
            data['y_train'] = y[pos[0:int(n * split_ratio[0])]]

            # data['x_valid'] = x1[pos[int(n * split_ratio[0]): int(n * split_ratio[0]) + int(n * split_ratio[1])], :]
            # data['y_valid'] = y[pos[int(n * split_ratio[0]): int(n * split_ratio[0]) + int(n * split_ratio[1])]]

            data['x_test'] = x1[pos[int(n * split_ratio[0]):], :]
            data['y_test'] = y[pos[int(n * split_ratio[0]):]]

            model = rc_model(input_shape=[seq_len, nb_feature])
            # model.summary()

            model_folder = os.path.join('processed_datasets/', data_opt)
            model_name = 'sp_{}_seqlen_{}'.format(sample, seq_len)
            model_train(model, file_name=os.path.join(model_folder, model_name), data=data, epochs=epochs,
                        batch_size=batch_size)
            # dependencies = {
            #     'auc_roc': roc_auc
            # }
            best_model = load_model(os.path.join(model_folder, model_name))

            rs = model_evaluate(best_model, x=data['x_test'], y=data['y_test'])
            f = open(os.path.join('processed_datasets/', data_opt + '.txt'), 'a')
            info = 'sp_{}_seqlen_{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(sample, seq_len,
                                                                                                      rs[0], rs[1],
                                                                                                      rs[2], rs[3],
                                                                                                      rs[4], rs[5],
                                                                                                      rs[6])
            f.write(info)
            f.close()
            print(info)

            for i in range(7):
                rs_avg[seq_len][i] += rs[i]

    for seq_len in seq_lens:
        for i in range(7):
            rs_avg[seq_len][i] /= nb_sample
        rs = rs_avg[seq_len]
        print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(rs[0], rs[1], rs[2], rs[3], rs[4], rs[5],
                                                                              rs[6]))


if __name__ == '__main__':
    main()
