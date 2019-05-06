"""
1. build model
2. train model on data
3. evaluation
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# todo datgaset quick start guide
BATCH_SIZE = 2
train_dataset = tf.data.experimental.make_csv_dataset(
        'iris_training.csv', BATCH_SIZE,
        column_names=column_names, label_name=label_name, num_epochs=1
        )
# for i, [input, target] in enumerate(train_dataset):
#     print('{:}th data, {:} with features {:}'.format(i, target, input))

plt.scatter