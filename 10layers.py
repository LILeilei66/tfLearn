"""
tf.keras.layers.Dense:
======================
    The layer has never been called and thus has no defined output || input shape.
    While calling this object, the arguments should of dimension of two, with the shape[0] the
    instance size, and shape[1] the feature size.
difference between tf.layers and tf.keras.layers:
==================================================
    tf.layers is the wrappers of tf.keras.layers, e.g.:
        @tf_export('layers.Dense')
        class Dense(keras_layers.Dense, base.Layer):
why len(tf.layers.Dense.weight) is 2?
=====================================
    <class 'list'>:
    [<tf.Variable 'dense/kernel:0' shape=(10, 100) dtype=float32, numpy=array(), dtype=float32)>,
    <tf.Variable 'dense/bias:0' shape=(100,) dtype=float32, numpy=array(), dtype=float32)>]
    答: 在 tf, weight 不叫 weight, 叫 kernel.
"""

import tensorflow as tf
tf.enable_eager_execution()

# <editor-fold desc="利用 input 确定input shape of layer">
layer = tf.keras.layers.Dense(100, input_shape=[None, 10]) # first argument


# signifies the output
# channel.

inputs = tf.random_normal(shape=[1,10], mean=0.0)
outputs = layer(inputs)


result = tf.matmul(inputs.numpy(), layer.kernel.numpy()) + layer.bias.numpy()
assert result.numpy().all() == outputs.numpy().all()
print(type(layer.bias))
# print(outputs)
print((layer.weights[0].shape))
inputs2 = tf.random_normal(shape=[100,10],mean=0.0)
outputs2 = layer(inputs2)
# </editor-fold>
