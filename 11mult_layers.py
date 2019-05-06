"""
目前发现可以完成卷积的 module 有:
tf.nn.conv2d || tf.layers.conv2d || tf.keras.layers.conv2d

conv2d
------
    Input shape:   # 似乎默认是 channels_last
        [samples, channels, rows, cols] || [samples, rows, cols, channels]
        取决于 dataformat=['channels_first', 'channels_last']

    kernel shape:
        [filter_height, filter_width, in_channels, out_channels]
        <@pytorch, weight=[out_channels, in_channels, filter_height, filter_width]>

    output shape:
        [samples, filters, new_rows, new_cols]
        || [samples, new_rows, new_cols, filters]
        取决于 dataformat=['channels_first', 'channels_last']


@知乎评价: https://zhuanlan.zhihu.com/p/45199737
-----------------------------------------------
    tf.nn: 最底层
    tf.layers: 用 tf.nn 造的轮子,
    tf.keras: 车, 基于 tf.layers 和 tf.nn 的高度封装.

    区别在于参数:
    tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu, dilations)
    tf.keras.layers.Conv2d || tf.layers.conv2d 没区别
    [官网建议使用 keras.layers.conv2d ]

    stackoverflow 说
    @tf.nn.conv2d: <建议用于 loading pretrained model>
        filter: tensor, same type as input
            => 4d of shape [filter_height, filter_width, in_channels, out_channels]
    @tf.layers.conv2d:  <建议用于 trained from scratch>
        filter: Integer, dimensionality of the output space
    tf 称卷积核为 filter.
"""

import tensorflow as tf
tf.enable_eager_execution()

class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filter):
        super(ResnetIdentityBlock, self).__init__()
        filters1, filters2, filters3 = filter

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1,1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def __call__(self, input_tesnsor, training=False):
        x = self.conv2a(input_tesnsor) # filter's shape = [1, 1, 3, 4]
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x) # filter's shape = [1, 1, 4, 2]
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x) # filter's shape = [1, 1, 2, 3]
        x =self.bn2c(x,training=training)
        x += input_tesnsor
        return tf.nn.relu(x)

class ResnetIdentityBlock_seq(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super().__init__()
        filters1, filters2, filters3 = filters
        conv2a = tf.keras.layers.Conv2D(filters1, kernel_size=(1,1))
        conv2b = tf.keras.layers.Conv2D(filters2, kernel_size=kernel_size)
        conv2c = tf.keras.layers.Conv2D(filters3, kernel_size=(1, 1))

        self.sequence = tf.keras.Sequential([conv2a,
                                       tf.keras.layers.BatchNormalization(),
                                       tf.keras.layers.ReLU(),
                                       conv2b,
                                       tf.keras.layers.BatchNormalization(),
                                       tf.keras.layers.ReLU(),
                                       conv2c,
                                       tf.keras.layers.BatchNormalization()])

    def __call__(self, input_tensor, training):
        x = self.sequence(input_tensor)
        x += input_tensor
        return tf.nn.relu(x)

block = ResnetIdentityBlock_seq(1,[4,2,3])
print(block(tf.zeros([1,2,2,3]), training=False))
print([x.name for x in block.trainable_variables])