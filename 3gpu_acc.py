import tensorflow as tf

tf.enable_eager_execution()

x = tf.random_uniform([3, 3])
print(tf.test.is_gpu_available())
print(x.device) # /job:localhost/replica:0/task:0/device:GPU:0

print(x.device.endswith('GPU:0'))

with tf.device('CPU:0'):
    print(x.device) # still GPU:0 nothing changed....