"""
Estimator 要求 input_fn 不带参

"""

import tensorflow as tf
import pandas as pd
import tensorflow.feature_column as fc
import os
import sys
import matplotlib.pyplot as plt
import functools
import census_dataset

tf.enable_eager_execution()

# <editor-fold desc="dataset creation">
TRAIN_FILE = 'adult.data'
TEST_FILE = 'adult.test'

train_df = pd.read_csv(TRAIN_FILE, header=None, names=census_dataset._CSV_COLUMNS)
test_df = pd.read_csv(TEST_FILE, header=None, names=census_dataset._CSV_COLUMNS)
# print(train_df.head())
# print(test_df.head())
# print(test_df.shape)

def easy_input_function(df, label_key, num_epochs, shuffle, batch_size):
    label = df[label_key]
    ds = tf.data.Dataset.from_tensor_slices((dict(df), label))

    if shuffle:
        ds = ds.shuffle(10000)

    ds = ds.batch(batch_size).repeat(num_epochs)

# </editor-fold>

def parse_csv(value):
    columns = tf.decode_csv(value, record_defaults=[0,0,0,0])
    features = dict(zip(['ft1', 'ft2', 'ft3', 'label'], columns)) # dict(k, v)
    labels = features.pop('label')
    return features, labels

def input_fn(file_names, num_epoch, shuffle, batch_size):
    ds = tf.data.TextLineDataset(file_names)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    ds = ds.map(parse_csv)
    ds = ds.repeat(num_epoch).batch(batch_size)
    return ds

train_inpf = functools.partial(input_fn, TRAIN_FILE, num_epochs=2, shuffle=True, batch_size=64)
test_inpf = functools.partial(input_fn, TEST_FILE, num_epochs=1, shuffle=True, batch_size=64)

