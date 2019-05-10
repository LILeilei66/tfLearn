"""
利用 fashion MNIST 数据集进行分类 task .
整体思路:
    1. load Dataset
    2. data pre-processing
    3. 模型创建及编译
    4. 模型训练
    5. 模型评估
    6. 进行预测
Data Description:
    training: 6e5
    testing: 1e5
ref: https://www.tensorflow.org/tutorials/keras/basic_classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.examples.tutorials.mnist import input_data
import gzip
import numpy as np
import matplotlib.pyplot as plt
import os

# tf.enable_eager_execution()

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# <editor-fold desc="1. Create Dataset">
def get_fashion_mnist_dataset(dirname = 'E:\\tfLearn\\18basic_classification', cache_subidr='data'):
    """
    当发现在 os.path.join(dirname, cache_subdir) 中存在所需文件时, 不会再进行下载.
    :param dirname:
    :param cache_subidr:
    :return:
    """
    base = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'

    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz']
    paths = []
    for fname in files:
        paths.append(get_file(fname, origin=base + fname, cache_dir=dirname, cache_subdir=cache_subidr))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

(train_images, train_labels), (test_images, test_labels) = get_fashion_mnist_dataset()

# <editor-fold desc="dataset description">
print('==dataset description=====')
print('train_images.shape: ', train_images.shape)
print("len(train_labels)", len(train_labels))
print('test_images.shape: ', test_images.shape)
print("len(test_labels)", len(test_labels))
# </editor-fold>

plt.imshow(train_images[0])
plt.colorbar()
# plt.show()
# </editor-fold>

# <editor-fold desc="2. convert img value to (0, 1)">
train_images = train_images / 255.0
test_images = test_images / 255.0

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(train_images[i])
# plt.show()
# </editor-fold>

# <editor-fold desc="3. 模型创建及编译">
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # (28， 28) -> 784
    keras.layers.Dense(128, activation=tf.nn.relu), # output shape = 128
    keras.layers.Dense(10, activation=tf.nn.softmax) # output shape = 10
    ])
keras.layers.MaxPool2D
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=[
    'accuracy'])
# </editor-fold>


# <editor-fold desc="4. 模型训练">
model.fit(x=train_images, y=train_labels, epochs=5) # hahh竟然是 model.fit, 这是 skleanr 么?
# </editor-fold>

# <editor-fold desc="5. 模型评估">
test_loss, test_acc = model.evaluate(x=test_images, y=test_labels)
print(test_loss, test_acc)
# </editor-fold>

# <editor-fold desc="6. 进行预测">
predictions = np.argmax(model.predict(test_images), axis=1)
print([predictions[:5], test_labels[:5]])
# </editor-fold>