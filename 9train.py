"""
当使用inputs of normal (with mean = 0.0) the result is amazing,
However, with mean != 0.0, the result is horrible.
Actually, the bias never converges to the expected value.

所以这里其实证明了, softmax 的意义.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()

TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

class Model(object):
    def __init__(self):
        self.W = tf.contrib.eager.Variable(5.0)
        self.b = tf.contrib.eager.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b

def loss(prediction, target):
    return tf.reduce_mean(tf.square(prediction - target))

def train(model, input, target, lr):
    print(lr)
    with tf.GradientTape() as g:
        g.watch([model.W, model.b])
        current_loss = loss(model(input), target)
    dW, db = g.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(lr * dW)
    model.b.assign_sub(lr * db)


if __name__ == '__main__':
    # <editor-fold desc="ground truth">
    inputs = tf.random_normal(shape=[NUM_EXAMPLES], mean=0.0)
    noises = tf.random_normal(shape=[NUM_EXAMPLES], mean=0.0)
    targets = TRUE_W * inputs + TRUE_b + noises
    # </editor-fold>

    Ws, bs, loss_list = [], [], []

    model = Model()
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    loss_list.append(loss(model(inputs), targets).numpy())
    epochs = 100
    for epoch in range(1, epochs):
        train(model, input=inputs, target=targets, lr=0.1 * 0.9 ** (epoch // 5))
        Ws.append(model.W.numpy())
        bs.append(model.b.numpy())
        loss_list.append(loss(model(inputs), targets).numpy())
        print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' \
              % (epoch, Ws[-1], bs[-1], loss_list[-1]))

    plt.plot(range(epochs), Ws, 'r'), plt.plot(range(epochs), bs, 'b')
    plt.plot([TRUE_W] * epochs, 'r--'), plt.plot([TRUE_b] * epochs, 'b--')
    plt.legend(['W', 'b', 'true W', 'true b'])
    plt.show()