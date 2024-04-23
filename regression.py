
import gc
import random as rn

import numpy as np
from sklearn.metrics import *
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Bidirectional, LSTM, Input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

import utils
import encoder
import constants

# 保证代码复现
rn.seed(7)
np.random.seed(7)
tf.random.set_seed(7)


def callbacks(ds_name, model_name):

    checkpoint = ModelCheckpoint(
        f'../result/{ds_name}/{model_name}/training' +
        '/{epoch:03d}-train_{loss:.4f}-tune_{val_loss:.4f}.h5',
        monitor='val_loss', verbose=0, save_best_only=True, mode='min', period=1)

    tboard_callback = TensorBoard(
        f'../result/{ds_name}/{model_name}/logs/', update_freq=5)

    earlystopping = EarlyStopping(monitor='val_loss', verbose=0, patience=15, mode='min', restore_best_weights=True)

    return [tboard_callback, checkpoint, earlystopping]


def fc_model(ds_name, data_shape):

    K.clear_session()

    model_name = 'fc'
    model_dict = {'avgfp': [100, 100, 100, 0.0001, 32], 'bgl3': [100, 100, 0.0001, 32], 'gb1': [1000, 0.0001, 64],
                  'pab1': [100, 100, 100, 0.001, 128], 'ube4b': [100, 100, 100, 0.0001, 64]}

    model = Sequential()
    model.add(Flatten(input_shape=(data_shape[1], data_shape[2])))

    for neural_count in model_dict[ds_name][:-2]:

        model.add(Dense(neural_count, activation=tf.nn.leaky_relu))
        model.add(Dropout(0.2))

    model.add(Dense(1))

    # adam optimizer一种优化方法。合理的动态选择每部梯度下降的幅度，控制学习速度
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=model_dict[ds_name][-2]))
    model.summary()

    model_batch = model_dict[ds_name][-1]

    return model, model_name, model_batch


def bilstm_model(data_shape):

    K.clear_session()

    model_name = 'bilstm'

    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=False), merge_mode='concat',
                            input_shape=(data_shape[1], data_shape[2])))
    model.add(Flatten())
    model.add(Dense(100, activation=tf.nn.leaky_relu))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # adam optimizer一种优化方法。合理的动态选择每部梯度下降的幅度，控制学习速度
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=0.0001))
    model.summary()

    model_batch = 32

    return model, model_name, model_batch


def attention_encoder_model(data_shape):

    K.clear_session()

    model_name = 'attention_encoder'

    inputs = Input(shape=(data_shape[1], 40))
    out_seq = encoder.Encoder(1, 40, 1, 64, data_shape[1])(inputs)
    flatten = Flatten()(out_seq)
    dense1 = Dense(100, activation=tf.nn.leaky_relu)(flatten)
    dropout1 = Dropout(0.2)(dense1)
    dense2 = Dense(100, activation=tf.nn.leaky_relu)(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    output = Dense(1)(dropout2)
    model = Model(inputs=inputs, outputs=output)

    # adam optimizer一种优化方法。合理的动态选择每部梯度下降的幅度，控制学习速度
    model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=0.0001))
    model.summary()

    model_batch = 32

    return model, model_name, model_batch


def main():

    for ds_name in constants.DATASETS.keys():

        data = utils.load_split_dataset(ds_name)

        model, model_name, model_batch = attention_encoder_model(data['X_train'].shape)

        # 模型训练
        model.fit(data['X_train'], data['y_train'], validation_data=(data['X_tune'], data['y_tune']), epochs=300,
                  batch_size=model_batch, verbose=1, callbacks=callbacks(ds_name, model_name))

        del data
        gc.collect()

        data = utils.load_split_dataset(ds_name, train=False)

        X_pred = model.predict(data['X_test'])

        from sklearn.metrics import mean_squared_error
        from scipy.stats import spearmanr, pearsonr

        print(mean_squared_error(X_pred, data['y_test']))
        print(spearmanr(X_pred.reshape(-1), data['y_test'].reshape(-1))[0])
        print(pearsonr(X_pred.reshape(-1), data['y_test'].reshape(-1)))


if __name__ == '__main__':

    main()

