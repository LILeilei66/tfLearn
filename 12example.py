"""
1. build model
2. train model on data
3. evaluation

tfe.metrics.Mean()
    (@Mean(Metric), def build(self, *args, **kwargs): del args, kwargs) 可还行...

    function: call:
        call(self, values, weights)
        关键操作为 assign_add.

    function: result:
        result(self)
        return self.numer / self.denom

    e.g.: call([1,3,5,7]) -> 4; call([1,3,5,7], [1,1,0,0]) -> 2

"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# todo dataaset quick start guide
BATCH_SIZE = 2

# <editor-fold desc="train dataset">
train_dataset = tf.contrib.data.make_csv_dataset(
        '12iris_training.csv', BATCH_SIZE,
        column_names=column_names, label_name=label_name, num_epochs=1
        )
# for i, [input, target] in enumerate(train_dataset):
#     print('{:}th data, {:} with features {:}'.format(i, target, input))

inputs, target = next(iter(train_dataset))

# need to pack the inputs into shape (batch_size, num_features)
def pack_features_vector(inputs, target):
    return tf.stack(list(inputs.values()), axis=1), target

train_dataset = train_dataset.map(pack_features_vector)
# </editor-fold>

# <editor-fold desc="test dataset">
test_dataset = tf.contrib.data.make_csv_dataset(
    '12iris_testing.csv', BATCH_SIZE,
    column_names=column_names, label_name=label_name, num_epochs=1
    )
test_dataset = test_dataset.map(pack_features_vector)
# </editor-fold>

# <editor-fold desc="draw features">
# plt.scatter(inputs['petal_length'], inputs['sepal_length'], c=target.numpy())
# plt.xlabel('petal_length'), plt.ylabel('sepal_length')
# plt.show()
# </editor-fold>

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
    ])

inputs, target = next(iter(train_dataset))

prediction = tf.argmax(tf.nn.softmax(model(inputs)), axis=1)

def loss(model, x, y):
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=model(x))
# print(loss(model, inputs, target))

def grad(model, inputs, targets):
    with tf.GradientTape() as g:
        g.watch(model.trainable_variables)
        loss_value = loss(model, inputs, targets)
    return loss_value, g.gradient(loss_value, model.trainable_variables)

# <editor-fold desc="optimizer">
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
global_step = tf.train.get_or_create_global_step() # todo: 不是很确定这个的意义
print(global_step)

loss_value, grads = grad(model,inputs, target)
optimizer.apply_gradients(zip(grads, model.variables), global_step) # todo 怎么还要自己计算梯度？
# </editor-fold>

# <editor-fold desc="train">
EPOCHS = 10
train_loss_results = []
train_acc_results = []
for epoch in range(EPOCHS):

    epoch_loss_avg = tfe.metrics.Mean()
    epoch_acc_avg = tfe.metrics.Accuracy()

    for features, target in iter(train_dataset):
        loss_value, grads = grad(model, features, target)
        optimizer.apply_gradients(zip(grads, model.variables), global_step)

        epoch_loss_avg(loss_value)
        epoch_acc_avg(tf.argmax(model(features), axis=1, output_type=tf.int32), target)

    train_loss_results.append(epoch_loss_avg.result())
    train_acc_results.append(epoch_acc_avg.result())

    print('Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}'.format(epoch, epoch_loss_avg.result(),
                                                                epoch_acc_avg.result()))
# </editor-fold>

# <editor-fold desc="view train">
fig, axes = plt.subplots(2, sharex=True)
fig.suptitle('Training Metrics')

axes[0].set_ylabel('Loss')
axes[0].plot(train_loss_results)

axes[1].set_ylabel('acc')
axes[1].plot(train_acc_results)

fig.show()
# </editor-fold>


test_acc=tfe.metrics.Accuracy()
test_loss=tfe.metrics.Mean()

for [inputs, targets] in iter(test_dataset):

    prediction = tf.argmax(tf.nn.softmax(model(inputs)), axis=1, output_type=tf.int32)
    test_loss(loss(model, inputs, targets))
    test_acc(prediction, target)

print('Result of training: Loss: {:.3f}, Accuracy: {:.3%}'.format(test_loss.result(),
                                                                  test_acc.result()))