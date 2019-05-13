"""
构造函数 tf.Variable() 通过 initial_value 确定变量的形状和类型.
变量通过 assign() 进行改变

ref: https://blog.csdn.net/xierhacker/article/details/53103979
"""
import tensorflow as tf
import numpy as np

# <editor-fold desc="1. 创建 Variable">
w = tf.Variable(initial_value=[[1,2],[3,4]], dtype=tf.float32)
w = tf.assign(w, [[2,2],[2,2]])
x = tf.Variable(initial_value=[[1,1],[1,1]], dtype=tf.float32)
y = tf.matmul(w, x)
print(y) # Tensor("MatMul:0", shape=(2, 2), dtype=float32)

init_op = tf.global_variables_initializer() # todo: what's this
# 根据stackoverflow: 所有的变量都必须先 explicitly initialized, 通过 initializer.

with tf.Session() as session:
    session.run(init_op)
    y = session.run(y)
    print(y) # [[4, 4], [4, 4]] || [[3. 3.], [7. 7.]]
    print(tf.get_default_graph().get_all_collection_keys())
    for key in tf.get_default_graph().get_all_collection_keys():
        print(tf.get_default_graph().get_collection(key))
    """
    ['variables', 'trainable_variables']
    [<tf.Variable 'Variable:0' shape=(2, 2) dtype=float32_ref>, <tf.Variable 'Variable_1:0' shape=(2, 2) dtype=float32_ref>]
    [<tf.Variable 'Variable:0' shape=(2, 2) dtype=float32_ref>, <tf.Variable 'Variable_1:0' shape=(2, 2) dtype=float32_ref>]

    """
# </editor-fold>

# <editor-fold desc="2. tf.constant">
print('2.')
a = tf.constant(1., name='a')
print(a.name)
b = tf.constant([2., 2.,], name='b')
print(b.name)

with tf.Session() as sess:
    res_a = sess.run(a)
    print('res_a:', res_a)
    print('res_a.type:', type(res_a))
# </editor-fold>

# <editor-fold desc="3. 变量与初始化的一些函数">
"""
1. tf.global_variables()
2. tf.local_variables()
3. tf.variables_initializer(var_list, name='init'
4. tf.global_variables_initializer()
5. tf.local_variables_initializer()
"""
# </editor-fold>

# <editor-fold desc="4. fetches">
"""
fetches 表示一种取的动作, 
"""
print('4.')
a = tf.constant(1, name='a')
b = tf.constant(2, name='b')
add = tf.add(a, b)

with tf.Session() as sess:
    result = sess.run(add) # 此处 add 便被称为 fetch.
    print(result)
with tf.Session() as sess:
    sess.run(add)
    print(add)
del a, b, add, result
# </editor-fold>

# <editor-fold desc="5. feeds">
"""
feed (与placeholder 一同操作) : 
"""
print('5.')
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as session:
    result = session.run(output, {input1: 2, input2: 3})
    print(result)

del input1, input2, output, result
# </editor-fold>