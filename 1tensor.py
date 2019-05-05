"""
tf.add tf.matmal, tf.linalg.inv etc

tf.enable_eager_execution():
------------------------------
动态图，启用 Eager Execution 会改变 TensorFlow 操作的行为方式 - 现在它们会立即评估并将值返回给 Python。


"""
import tensorflow as tf

tf.enable_eager_execution()

print(tf.add(1,2)) # tf.Tensor(3, shape=(), dtype=int32)
print(tf.add([1,2], [3,4])) # tf.Tensor([4 6], shape=(2,), dtype=int32)
print(tf.square(5))  # tf.Tensor(25, shape=(), dtype=int32)
print(tf.reduce_sum([1,2,3])) # tf.Tensor(6, shape=(), dtype=int32)
print(tf.encode_base64("helo")) # tf.Tensor(b'aGVsbw', shape=(), dtype=string)
print(tf.square(2) + tf.square(3)) # tf.Tensor(13, shape=(), dtype=int32)

assert tf.add(1,2).numpy() == 3

x = tf.matmul([[1]], [[2,3]])
print(x.shape) # (1, 2)
print(x.dtype) # <dtype: 'int32'>

# <editor-fold desc="without enable_eager_execution">
"""
Tensor("Add:0", shape=(), dtype=int32)
Tensor("Add_1:0", shape=(2,), dtype=int32)
Tensor("Square:0", shape=(), dtype=int32)
Tensor("Sum:0", shape=(), dtype=int32)
Tensor("EncodeBase64:0", shape=(), dtype=string)
Tensor("add_2:0", shape=(), dtype=int32)
AttributeError
"""
# </editor-fold>