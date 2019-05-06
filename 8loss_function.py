import tensorflow as tf
import matplotlib.pyplot as plt
tf.enable_eager_execution()

W = 3.0
B = 2.0
NUM_EXAMPLES = 1000



class Model(object):
    def __init__(self):
        self.W = tf.contrib.eager.Variable(5.0)
        self.b = tf.contrib.eager.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b

def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

model = Model()
inputs = tf.random_uniform(shape=[NUM_EXAMPLES], minval=10)
noises = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * W + B + noises
print(loss(predicted_y=model(inputs), desired_y=outputs).numpy())