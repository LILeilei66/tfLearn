import tensorflow as tf
import pandas as pd
import tensorflow.feature_column as fc
import os
import sys
import matplotlib.pyplot as plt
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


# </editor-fold>



