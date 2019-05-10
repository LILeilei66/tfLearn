"""
对应 17feature_column.md
学习九大 feature_column functions:
    1. tf.feature_column.numeric_column;
"""
# todo:还是不知道怎么用啊?
import tensorflow as tf
# <editor-fold desc="1. tf.feature_column.numeric_column">
tf.feature_column.numeric_column()
# </editor-fold>

identity_feature_column = tf.feature_column.categorical_column_with_identity(key='fta',
                                                                             num_buckets=4)
def input_fn():
    return ({'fta': [1,2,3,0], 'ftb': [5,6,7,8]},
            [0,2,3,4])

identity_feature_column()
tf.estimator.LinearRegressor()