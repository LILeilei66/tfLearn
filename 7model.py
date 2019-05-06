"""
在tutorial中, 是直接使用tf.Variable, 但是我在用的时候始终报错, 只有 tf.contrib.eager.Variable可行.
"""

import tensorflow as tf
tf.enable_eager_execution()

v = tf.contrib.eager.Variable(1.0)
assert v.numpy() == 1.0

class Model(object):
    def __init__(self):
        self.W = tf.contrib.eager.Variable(5.0)
        self.b = tf.contrib.eager.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b

model = Model()
assert model(3.0).numpy() == 15.0
