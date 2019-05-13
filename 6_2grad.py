"""
ref: https://blog.csdn.net/xierhacker/article/details/53174558
"""
import tensorflow as tf

# <editor-fold desc="1. 一元求导">
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y1 = 2 * x
    y2 = x ** 2 + 2
    y3 = x ** 2 + x * x

dy1_dx  = tape.gradient(target=y1, sources=x)
dy2_dx  = tape.gradient(target=y2, sources=x)
dy3_dx  = tape.gradient(target=y3, sources=x)

with tf.Session() as sess:
    dy1_dx = sess.run(dy1_dx)

print('dy1_dx', dy1_dx) # dy1_dx 2.0
print('dy2_dx', dy2_dx) # dy2_dx Tensor("Reshape:0", shape=(), dtype=float32)
print('dy3_dx', dy3_dx) # dy3_dx Tensor("AddN:0", shape=(), dtype=float32)

# </editor-fold>

# <editor-fold desc="2. Layer 求导">
x = tf.constant([[2., 3.]])
a = tf.keras.layers.Dense(2)
b = tf.keras.layers.Dense(2)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    result = b(a(x))
    g_a = tape.gradient(a(x), x)
    g_b = tape.gradient(result, x)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    # v_a = sess.run(a.get_weights())
    # v_b = sess.run(b.get_weights())
    output_a = sess.run(a(x))
    result = sess.run(result)
    g_a = sess.run(g_a)
    g_b = sess.run(g_b)

print('v_a', a.get_weights())
"""v_a [array([[-0.40425915, -0.59929955],
       [-0.95141447,  0.4541161 ]], dtype=float32), array([0., 0.], dtype=float32)]"""
print('v_b', b.get_weights())
"""v_b [array([[-1.2015895 ,  0.3204056 ],
       [ 0.47647798,  0.54431033]], dtype=float32), array([0., 0.], dtype=float32)]"""
print('output_a', output_a) # output_a output_a [[ 5.505309  -1.9712033]]
print('result', result) # result [[-0.77420396 -5.688115  ]]
print('g_a', g_a) # g_a [[1.253601   0.34230137]]
print('g_b', g_b) # g_b [[-0.45510265 -1.8507048 ]]


# </editor-fold>