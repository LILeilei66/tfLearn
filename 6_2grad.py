"""
ref: https://blog.csdn.net/xierhacker/article/details/53174558
"""
import tensorflow as tf

# <editor-fold desc="1. 一元求导">
def func1():
    print('1.')
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

# <editor-fold desc="1.2. 矩阵求导">
"""
stack overflow:  gradien 目前是针对 scalar 的.
ref: https://stackoverflow.com/questions/48878053/tensorflow-gradient-with-respect-to-matrix
"""
def func12():

    print('1.2')
    x = tf.Variable(initial_value=[[1, 3]])
    w = tf.constant([[2]])

    with tf.GradientTape() as tape:
        tape.watch(w)
        res = tf.matmul(w, x)
    dw = tape.gradient(target=res, sources=w) # get None Type

    # init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        # sess.run(init_op)
        dw = sess.run(dw)
        # w = sess.run(w)

    print(dw)


# </editor-fold>

# <editor-fold desc="2. Layer 求导">
# todo: 始终没有成功提取 weight
def func2():
    print('2.')
    x = tf.constant([[1.]])
    a = tf.keras.layers.Dense(2)
    b = tf.keras.layers.Dense(2)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        res_b = b(a(x))
        res_a = a(x)
    g_a = tape.gradient(res_a, x)
    g_b = tape.gradient(res_b, x)

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        # v_a = sess.run(a.get_weights())
        # v_b = sess.run(b.get_weights())
        res_b = sess.run(res_b)
        res_a = sess.run(res_a)
        g_a = sess.run(g_a)
        g_b = sess.run(g_b)

    print('x', x)
    print('v_a', a.kernel)
    # print('')
    """v_a [array([[-0.40425915, -0.59929955],
           [-0.95141447,  0.4541161 ]], dtype=float32), array([0., 0.], dtype=float32)]"""
    print('v_b', b.get_weights())
    """v_b [array([[-1.2015895 ,  0.3204056 ],
           [ 0.47647798,  0.54431033]], dtype=float32), array([0., 0.], dtype=float32)]"""
    print('output_a', res_a) # output_a output_a [[ 5.505309  -1.9712033]]
    print('result', res_b) # result [[-0.77420396 -5.688115  ]]
    print('g_a', g_a) # g_a [[1.253601   0.34230137]]
    print('g_b', g_b) # g_b [[-0.45510265 -1.8507048 ]]

# </editor-fold>

func2()