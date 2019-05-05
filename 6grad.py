"""
tf.GradientTape:
================
Record operations for automatic differentiation.
parameters:
    persistent: bool, if True, gradient function can be called plural times.
    watch_accessed_variables: bool, if False, need to trace the expected tensor by hand.

watch():
========
Ensures that `tensor` is being traced by this tape. 确保某个 tensor 被 tape 追踪.

"""
import tensorflow as tf
tf.enable_eager_execution()

# <editor-fold desc="ex1">
x = tf.constant(3.0)
with tf.GradientTape() as g:
    g.watch(x)
    y = x * x
dy_dx = g.gradient(y, x)
print(dy_dx) # tf.Tensor(6.0, shape=(), dtype=float32)
# </editor-fold>

x1 = tf.constant(value=2.0)
x2 = tf.constant(value=3.0)
with tf.GradientTape() as g:
    g.watch([x1, x2])
    with tf.GradientTape() as gg:
        gg.watch([x1, x2])
        y = x1 * x1 + x2 * x2
    dy_dx = gg.gradient(y,sources=[x1, x2])
d2y_d2x = g.gradient(dy_dx, sources=[x1,x2])
print(dy_dx, d2y_d2x) # tf.Tensor(6.0, shape=(), dtype=float32) tf.Tensor(2.0, shape=(), dtype=float32)
# [<tf.Tensor: id=16, shape=(), dtype=float32, numpy=4.0>, <tf.Tensor: id=17, shape=(), dtype=float32, numpy=6.0>]
# [<tf.Tensor: id=26, shape=(), dtype=float32, numpy=2.0>, <tf.Tensor: id=27, shape=(), dtype=float32, numpy=2.0>]
# todo: 为什么只有两个不是三个？


with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    y = x * x
    z = x * 2
dy_dx = g.gradient(target=y, sources=x)
dz_dx = g.gradient(target=z, sources=x)
print(dy_dx, dz_dx) # tf.Tensor(6.0, shape=(), dtype=float32) tf.Tensor(2.0, shape=(), dtype=float32)
