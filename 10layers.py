"""
tf.keras.layers.Dense:
======================
    The layer has never been called and thus has no defined output || input shape.
    While calling this object, the arguments should of dimension of two, with the shape[0] the
    instance size, and shape[1] the feature size.


"""

import tensorflow as tf
tf.enable_eager_execution()

# <editor-fold desc="利用 input 确定input shape of layer">
layer = tf.keras.layers.Dense(100) # first argument signifies the output channel.
print(len(layer.weights))
inputs = tf.random_normal(shape=[10,10], mean=0.0)
outputs = layer(inputs)
print(outputs)
print((layer.weights[0].shape))
inputs2 = tf.random_normal(shape=[100,10],mean=0.0)
outputs2 = layer(inputs2)
# </editor-fold>