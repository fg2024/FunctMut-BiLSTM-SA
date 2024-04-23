import os
import random as rn

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import constants


# 保证代码复现
rn.seed(7)
np.random.seed(7)
tf.random.set_seed(7)


def split_dataset():

    for ds_name in constants.DATASETS.keys():

        try:
            os.mkdir(constants.IndexSplitDatasetPath.format(ds_name))
        except FileExistsError:
            pass

        for random_state in [6, 7, 8]:

            data = pd.read_csv(constants.SourceDatasetPath.format(ds_name, ds_name), sep="\t", header=0)

            index = np.arange(data.shape[0])
            train, test = train_test_split(index, test_size=0.1, random_state=random_state)
            train, tune = train_test_split(train, test_size=0.1, random_state=random_state)

            try:
                os.mkdir(constants.IndexSplitDatasetPath.format(ds_name) + 'random_state_{}/'.format(random_state))
            except FileExistsError:
                pass

            for ds_split_type, ds_split in zip(['train', 'tune', 'test'], [train, tune, test]):
                np.save(constants.IndexSplitDatasetPath.format(ds_name) +
                        'random_state_{}/{}.npy'.format(random_state, ds_split_type), ds_split)


def split_dataset_keep_stest():

    for ds_name in constants.DATASETS.keys():

        print(ds_name)

        for random_state in [6, 7, 8]:

            data = pd.read_csv(constants.SourceDatasetPath.format(ds_name, ds_name), sep="\t", header=0)

            data_train = pd.read_csv(constants.IndexSplitDatasetPath.format(ds_name) + 'train.txt', header=None).values.reshape(-1)

            data_tune = pd.read_csv(constants.IndexSplitDatasetPath.format(ds_name) + 'tune.txt', header=None).values.reshape(-1)

            data_test = pd.read_csv(constants.IndexSplitDatasetPath.format(ds_name) + 'stest.txt', header=None).values.reshape(-1)

            index = np.concatenate([data_train, data_tune], axis=0)

            data_train, data_tune = train_test_split(index, test_size=0.1, random_state=random_state)

            for ds_split_type, ds_split in zip(['train', 'tune', 'test'], [train, tune, test]):
                np.save(constants.IndexSplitDatasetPath.format(ds_name) +
                        'random_state_{}/{}.npy'.format(random_state, ds_split_type), ds_split)




def load_encode_dataset(ds_name=None, ec_name=None):

    if ds_name is None and ec_name is None:
        raise ValueError("must provide both ds_name and ec_name to load encode datasets")

    if isinstance(ec_name, str):

        if ec_name in constants.ENCODE_NAME:
            ec_name = [ec_name]
        else:
            raise ValueError("the provided encode method must be selected from aa_index and one_hot temporarily")

    if not isinstance(ec_name, list):
        raise ValueError("ec_name must be provided in list format if you need two or more encoded datasets")
    else:
        ec_data = []

        for ind, ec in enumerate(ec_name):

            ec_data.append(np.load(constants.EncodeDatasetPath.format(ds_name, ds_name, ec)).astype(float))

    return np.concatenate(ec_data, -1)


def load_split_dataset(ds_name, train=True):

    col_name = 'score'

    encode_data = load_encode_dataset(ds_name, ['one_hot', 'aa_index'])

    target_data = pd.read_csv(constants.SourceDatasetPath.format(ds_name, ds_name), sep="\t", header=0).loc[:, col_name].values.reshape(-1, 1)

    train_index, tune_index, test_index = \
        pd.read_csv('../data/' + ds_name + '/split/train.txt', header=None).values.reshape(-1),\
        pd.read_csv('../data/' + ds_name + '/split/tune.txt', header=None).values.reshape(-1),\
        pd.read_csv('../data/' + ds_name + '/split/stest.txt', header=None).values.reshape(-1)

    if train:
        return {
                'X_train': encode_data[train_index], 'X_tune': encode_data[tune_index],
                'y_train': target_data[train_index], 'y_tune': target_data[tune_index],
                'train_index': train_index,
                'tune_index': tune_index,
            }
    else:
        return {
                'X_test': encode_data[test_index],
                'y_test': target_data[test_index],
                'test_index': test_index
            }


if __name__ == '__main__':

    split_dataset_keep_stest()

    # split_dataset()

    # load_encode_dataset('avgfp', ['aa_index', 'one_hot'])

    # load_split_dataset('avgfp', 6)
