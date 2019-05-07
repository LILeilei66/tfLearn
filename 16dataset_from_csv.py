"""
tf.data.experimental.CsvDataset || tf.data.experimental.make_csv_dataset:
    make_csv_dataset 期中利用 CsvDataset 进行了 dataset 创建.

"""

import tensorflow as tf
tf.enable_eager_execution()
TRAIN_FP = 'data/fake_dataset.csv'

# <editor-fold desc="pandas read csv">
# method1: 利用 pandas 读取, 继而转成 tf 的 dataset
import pandas as pd
df = pd.read_csv(TRAIN_FP, header=None)
print(df)

features = df[df.columns[:-1]]
label = df[df.columns[-1]]
dataset = tf.data.Dataset.from_tensor_slices((dict(features), label))
print(dataset)
# </editor-fold>

# <editor-fold desc="tf read csv">
# method2: 直接利用 tf 读取 csv
def parse_csv(value):
    columns = tf.decode_csv(value, record_defaults=[0,0,0,0])
    target_column_names = ['ft0', 'ft1', 'ft2', 'label']
    features = dict(zip(target_column_names, columns))
    labels = features.pop('label')
    return features, labels

ds = tf.data.TextLineDataset(filenames=TRAIN_FP) # 继承于 dataset_ops.DatasetV1Adapter
ds = ds.map(parse_csv) # todo: 在这里的时候还是 <TextLineDatasetV1 shapes: (), types: tf.string>, map的时候就变成了Tensor("arg0:0", shape=(), dtype=string)
# 代码里面说关键在于 Dataset.map()
# ds = tf.decode_csv(ds._dataset.numpy(), record_defaults=[0,0,0,0])
#    records: A `Tensor` of type `string`.
print(ds)
# </editor-fold>

