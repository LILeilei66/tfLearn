"""
关键函数:
    apply_gradients(grads_and_vars):
        将梯度更新到变量上.
    tf.keras.losses 中找损失函数

咩儿:
    opt = tf.train.AdamOptimizer(lr, beta1, beta2)

    grad, _ = zip(*opt.compute_gradients(loss, theta_eg))
    grad, _ = tf.clip_by_global_norm(grad, grad_clip) # todo: what is tf.clip_by_global_norm and why here

    self.optimize_tot = opt.apply_gradients(zip(grad, theta_eg))
    self.optimize_rec = opt.minimize(self.loss_Rec, var_list=theta_eg）
    # todo: what is opt.minimize and why here?

ref: https://blog.csdn.net/xierhacker/article/details/53174558
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# todo 未完成, 因为看大家都不用 GradientTape 所以不太明白这个部分，看是否能找到关于 train.Optimizer 的相关内容.
EPOCHS = 2

train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10
# plt.scatter(train_X, train_Y), plt.show()

w = tf.Variable(initial_value=1.0)
b = tf.Variable(initial_value=1.0)

optimizer = tf.keras.optimizers.SGD(lr=0.01)
crit = tf.keras.losses.MeanSquaredError()

loss_list = []
w_list = []
b_list = []
for epoch in range(EPOCHS):
    print('---{:} epoch---'.format(epoch))

    """with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run([w, b])
        print("w:", w.eval())
        print("b:", b.eval())
        w_list.append(w.eval())
        b_list.append(b.eval())"""

    with tf.GradientTape() as tape:
        pred = w * train_X + b
        loss = crit(train_Y, pred)

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(loss)
    #     print('loss:', loss.eval())
    #     loss_list.append(loss.eval()) # todo: 这样么？

    with tf.Session() as sess:
        grad = tape.gradient(target=loss, sources=[tf.convert_to_tensor(w), tf.convert_to_tensor(b)])
        optimizer.apply_gradients(zip(grad, [w, b]))
        print('grad:', grad.eval())
