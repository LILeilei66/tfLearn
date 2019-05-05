"""
@pytorch:
cuda tensor needs to be converted to cpu then able to convert to numpy,

@tf:
gputensors can be converted to numpy directly.
"""

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

a = np.ones((3, 3))
t_a = tf.multiply(a, 42)
print(t_a)
# <editor-fold desc="result">
"""
tf.Tensor(
[[42. 42. 42.]
 [42. 42. 42.]
 [42. 42. 42.]], shape=(3, 3), dtype=float64)
"""
# </editor-fold>

print(np.add(t_a, 1)) # of class np
print(t_a.device) # /job:localhost/replica:0/task:0/device:GPU:0
# <editor-fold desc="result">
"""
[[43. 43. 43.]
 [43. 43. 43.]
 [43. 43. 43.]]
"""
# </editor-fold>

import torch
t_b = torch.tensor([1,2]).to('cuda:0')
print(t_b.device) # cpu
b = np.add(t_b, 1) # of class tensor
print(b)