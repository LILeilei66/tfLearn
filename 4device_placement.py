"""
CPU non-availability:
Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
-----------------------------------------------------------------------------------------
AVX : Advanced Vector Extensions
用以进行矩阵乘法计算.
但是一般tf没有这个compile, todo: hgf他們是直接用有gpu的电脑使用的？他们不是笔记本么？
"""
import tensorflow as tf
import time

def time_matmal(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)

    result = time.time() - start
    print("{:0.2f}".format(1000 * result))

print('on cpu')
with tf.device("CPU:0"):
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmal(x)


time.sleep(5)
print('on gpu')
if tf.test.is_gpu_available():
    with tf.device("GPU:0"):
        x = tf.random_uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matmal(x)