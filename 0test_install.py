import tensorflow as tf
print(tf.__version__)
a = tf.random_normal((100, 100))
b = tf.random_normal((100, 500))
c = tf.matmul(a, b)
sess = tf.InteractiveSession()
sess.run(c)
