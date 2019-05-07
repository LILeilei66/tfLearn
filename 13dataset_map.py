"""
在12中有一个步骤是将 dataset 重新 pack, 具体做法是先创建 dataset 和 pack function; 继而在 dataset 上 map pack function.
"""
import tensorflow as tf
tf.enable_eager_execution()

a = tf.data.Dataset.from_tensor_slices([1,2,3,4,5])
b = a.map(lambda x: x + 1)
for value in iter(b):
    print(value)
"""
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(5, shape=(), dtype=int32)
tf.Tensor(6, shape=(), dtype=int32)
"""