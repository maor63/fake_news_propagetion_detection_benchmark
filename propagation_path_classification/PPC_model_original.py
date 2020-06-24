import sys, os, json, time, datetime

import numpy as np
import random

import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Concatenate, TimeDistributed, LSTM, \
    AveragePooling1D, Embedding, GRU, GlobalAveragePooling1D
from keras import initializers, regularizers
from keras.initializers import RandomNormal
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

conv_len = 5
cut_off = 0.5


def rc_model(input_shape, output_size=2):
    input_f = Input(shape=(input_shape[0], input_shape[1],), dtype='float32', name='input_f')
    r = GRU(32, return_sequences=True)(input_f)
    r = GlobalAveragePooling1D()(r)

    c = Conv1D(32, conv_len, activation='relu')(input_f)
    # c = Conv1D(64, conv_len, activation='relu')(c)
    c = MaxPooling1D(3)(c)
    c = GlobalAveragePooling1D()(c)

    rc = Concatenate()([r, c])
    rc = Dense(32, activation='relu')(rc)
    rc = Dropout(0.5)(rc)
    output_f = Dense(output_size, activation='softmax', name='output_f')(rc)
    model = Model(inputs=[input_f], outputs=[output_f])
    return model


def model_train(model, file_name, data, epochs, batch_size):
    model.compile(loss={'output_f': 'categorical_crossentropy'}, optimizer='rmsprop', metrics=['accuracy'])
    call_back = ModelCheckpoint(file_name, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False,
                                mode='auto', period=1)
    input_train = {'input_f': data['x_train']}
    output_train = {'output_f': data['y_train']}
    # input_valid = {'input_f': data['x_valid']}
    # output_valid = {'output_f': data['y_valid']}
    model.fit(input_train, output_train, epochs=epochs, batch_size=batch_size,
              # validation_data=(input_valid, output_valid),
              validation_split=0.25,
              callbacks=[call_back], verbose=2)


def model_predict(model, x):
    input_test = {'input_f': x}
    pred_test = model.predict(input_test)
    pred_test = pred_test.reshape((pred_test.shape[0],))
    return pred_test


def compute_metrics(pred_test, y_test):
    auc = roc_auc_score(y_test, pred_test)
    lp = np.argmax(pred_test, axis=1)
    lt = np.argmax(y_test, axis=1)
    acc = accuracy_score(lt, lp)
    f1s = []
    for pos_label in range(np.unique(lt).size):
        f1s.append(f1_score(lp, lt, labels=[pos_label], average='micro'))

    res = [acc, *f1s, auc]

    return res


def normalize(x, positions):
    num_columns = x.shape[1]
    for i in range(num_columns):
        if i in positions:
            x[:, i:i + 1] = np.copy(preprocessing.robust_scale(x[:, i:i + 1]))
    return x


def model_evaluate(model, x, y):
    input_test = {'input_f': x}
    pred_test = model.predict(input_test)
    y_test = y
    rs = compute_metrics(pred_test, y_test)
    return rs


def main():
    project_folder = 'datasets/twitter15/'
    # seq_len = 35
    epochs = 200
    batch_size = 128
    nb_sample = 10
    seq_lens = [100]
    data_opt = 'twitter16'
    output_size = 4

    X = np.load(f'processed_datasets/{data_opt}/X.npy')
    y_origin = np.load(f'processed_datasets/{data_opt}/y.npy')

    y = to_categorical(y_origin)
    # print(x.shape, y.shape)

    n = X.shape[0]
    nb_feature = X.shape[2]
    x = X.astype('float32')
    pos = np.arange(n)

    rs_avg = {}
    for seq_len in seq_lens:
        rs_avg[seq_len] = [0 for i in range(7)]

    skf = StratifiedKFold(n_splits=10)

    shape = x.shape
    x = x.reshape([shape[0] * shape[1], shape[2]])

    if 'twitter' in data_opt or 'fake_news' in data_opt:
        pos_norm = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    else:
        pos_norm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    x = normalize(x, pos_norm)
    x = x.reshape([shape[0], shape[1], shape[2]])

    for sample, (train_index, test_index) in enumerate(skf.split(x, y_origin)):
        print('sample {}'.format(sample))

        for seq_len in seq_lens:
            data = {}

            data['x_train'] = x[train_index, 0:seq_len, :]
            data['y_train'] = y[train_index]

            data['x_test'] = x[test_index, 0:seq_len, :]
            data['y_test'] = y[test_index]

            model = rc_model(input_shape=[seq_len, nb_feature], output_size=output_size)
            # model.summary()

            model_folder = os.path.join('processed_datasets/', data_opt)
            model_name = 'sp_{}_seqlen_{}'.format(sample, seq_len)
            model_train(model, file_name=os.path.join(model_folder, model_name), data=data, epochs=epochs,
                        batch_size=batch_size)
            best_model = load_model(os.path.join(model_folder, model_name))

            rs = model_evaluate(best_model, x=data['x_test'], y=data['y_test'])
            result_folder = f'processed_datasets/{data_opt}'
            if not os.path.exists(os.path.join(result_folder, 'result')):
                os.makedirs(os.path.join(result_folder, 'result'))
            f = open(os.path.join(result_folder, 'result', data_opt + '.txt'), 'a')
            info = 'sp_{}_seqlen_{}\t{}\n'.format(sample, seq_len, '\t'.join(map(lambda x: '{:.4f}'.format(x), rs)))
            f.write(info)
            f.close()
            print(info)

            for i in range(len(rs)):
                rs_avg[seq_len][i] += rs[i]

    for seq_len in seq_lens:
        for i in range(len(rs_avg[seq_len])):
            rs_avg[seq_len][i] /= nb_sample
        rs = rs_avg[seq_len]
        print('\t'.join(map(lambda x: '{:.4f}'.format(x), rs)))


if __name__ == '__main__':
    main()
