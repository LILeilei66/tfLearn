"""
似乎 iter dataset 与 take(1) 都可以起到取 batch 的作用，但是他们的返回值不同:
    ds.take(1) -> DatasetV1Adapter, 所以若希望 get features 与 labels, 仍需 iter.
    next(iter(ds)) -> tuple
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import pandas as pd
tf.enable_eager_execution()

TRAIN_FP = 'data/fake_dataset.csv'

df = pd.read_csv(TRAIN_FP, header=None)
label = df[df.columns[-1]]
features = df[df.columns[:-1]]

ds = tf.data.Dataset.from_tensor_slices((dict(features), label))
ds = ds.batch(1)

# method1: ds.take:
batch1 = ds.take(1)
for feature1, label1 in iter(batch1):
    print(feature1, label1)
print(batch1)

# method2: iterations
batch2 = next(iter(ds))
print(batch2)

# method3: tfe.Iterator(dataset)
for element in tfe.Iterator(ds):
    print(element)