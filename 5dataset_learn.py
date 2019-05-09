"""
tf.data.TextLineDataset 的相关 transformation 使用顺序:
    注意: 先 shuffle 再 map, 先 repeat 再 batch.
    ds = tf.data.TextLineDataset(file)
    ds = ds.shuffle(buffer_size)
    ds = ds.map(map_func)
    ds = ds.repeat(num_epoch).batch(batch_size)


tfe contains eager_execution and graph_execution which are efficient for image processing.
Dataset.from_tensors
Dataset.from_tensor_slices
tf.data.Dataset
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import tempfile
tf.enable_eager_execution()

# <editor-fold desc="tensor slices">
ds_tensors = tf.data.Dataset.from_tensor_slices([1,2,3,4])
print(ds_tensors) # <TensorSliceDataset shapes: (), types: tf.int32>

for tensor in enumerate(ds_tensors):
    print(tensor)
"""
(0, <tf.Tensor: id=4, shape=(), dtype=int32, numpy=1>)
(1, <tf.Tensor: id=6, shape=(), dtype=int32, numpy=2>)
(2, <tf.Tensor: id=8, shape=(), dtype=int32, numpy=3>)
(3, <tf.Tensor: id=10, shape=(), dtype=int32, numpy=4>)
"""
for tensor in ds_tensors:
    print(tensor)
"""
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
"""
# </editor-fold>

# <editor-fold desc="Text Line">
_, filename = tempfile.mkstemp()
with open(filename, 'w') as f:
    f.write("""Line1
Line2""")
ds_file = tf.data.TextLineDataset(filename)
print(ds_file) # <TextLineDataset shapes: (), types: tf.string>
for line in ds_file:
    print(line)
"""
tf.Tensor(b'Line1', shape=(), dtype=string)
tf.Tensor(b'    Line2', shape=(), dtype=string)
"""
# </editor-fold>

# dataset with dictionary
dataset = tf.data.Dataset.from_tensor_slices(
    {
        "img": np.array([1,2,3,4,5]),
        "label": np.random.uniform(siz=(5,2))
        }
    )
for one_element in tfe.Iterator(dataset):
    print(one_element)

# dataset with tuple
dataset = tf.data.Dataset.from_tensor_slices(
    (np.array([1,2,3,4,5]), np.random.uniform(siz=(5,2)))
    )