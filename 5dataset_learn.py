"""
Dataset.from_tensors
Dataset.from_tensor_slices
tf.data.Dataset
"""

import tensorflow as tf
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

