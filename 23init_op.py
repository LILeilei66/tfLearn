"""
问题:
    FailedPreconditionError (see above for traceback): Attempting to use uninitialized value Variable_1

解答:
    所有的 Variable 在 sess.run() 之前需要被 init;

解决方式:
    init_op = tf.initialize_all_variables()

ref: https://stackoverflow.com/questions/34001922/failedpreconditionerror-attempting-to-use-uninitialized-in-tensorflow
"""

import tensorflow as tf

a = tf.Variable(initial_value=[2.0])
b = tf.Variable(initial_value=[5.0])
c = tf.multiply(a, b)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op) # 少了就error.
    c = sess.run(c)

print(c) # [10.]

