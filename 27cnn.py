"""
根据 declaration:
    tf.nn.conv2d(input, filter, strides, padding,
                 use_cudnn_on_gpu=True, data_format="NHWC",
                 dilations=[1, 1, 1, 1], name=None)

    input of shape: [batch, in_height, in_width, in_channels]
    filter of shape: [filter_height, filter_width, in_channels, out_channels]



ref: https://blog.csdn.net/xierhacker/article/details/53174594
"""

import numpy as np
import tensorflow as tf

# <editor-fold desc="0. data preparation">
x=np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
            [[0,0,0],[0,1,2],[1,1,0],[1,1,2],[2,2,0],[2,0,2],[0,0,0]],
            [[0,0,0],[0,0,0],[1,2,0],[1,1,1],[0,1,2],[0,2,1],[0,0,0]],
            [[0,0,0],[1,1,1],[1,2,0],[0,0,2],[1,0,2],[0,2,1],[0,0,0]],
            [[0,0,0],[1,0,2],[0,2,0],[1,1,2],[1,2,0],[1,1,0],[0,0,0]],
            [[0,0,0],[0,2,0],[2,0,0],[0,1,1],[1,2,1],[0,0,2],[0,0,0]],
            [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]])
# of shape (7, 7, 3)

W=np.array([[[[1,-1,0],[1,0,1],[-1,-1,0]],
             [[-1,0,1],[0,0,0],[1,-1,1]],
             [[-1,1,0],[-1,-1,-1],[0,0,1]]],

            [[[-1,1,-1],[-1,-1,0],[0,0,1]],
             [[-1,-1,1],[1,0,0],[0,-1,1]],
             [[-1,-1,0],[1,0,-1],[0,0,0]]]])
# of shape (2, 3, 3, 3)

b = np.array([1, 0])

x = x[np.newaxis, :] # of shape (1, 7, 7, 3)
W = np.transpose(W, [1,2,3,0]) # of shape (3,3,3,2)

# </editor-fold>

# <editor-fold desc="1. graph creation">
g = tf.Graph()
with g.as_default():
    input = tf.constant(value=x, dtype=tf.float32, name='input')
    filter = tf.constant(value=W, dtype=tf.float32, name='filter')
    bias = tf.constant(value=b, dtype=tf.float32, name='bias')
    result = tf.nn.conv2d(input, filter, strides=[1,2,2,1], padding='SAME') + bias
    # of shape (1, 4, 4, 2)
    # + bias 得到: [:,:,:,0] + 1; [:,:,:,1] + 0
# </editor-fold>

# <editor-fold desc="2. session start">
with tf.Session(graph=g) as sess:
    r=sess.run(result)
    print(r)
# </editor-fold>