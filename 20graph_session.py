"""
ref: https://blog.csdn.net/xierhacker/article/details/53860379
"""
import tensorflow as tf
import numpy as np

# <editor-fold desc="1. 默认图">
"""
随着任务的开始, 就有了默认图, 通过 tf.get_default_graph() 进行访问.
"""
c = tf.constant(value=1)
print('c:', c.graph)
print('def:', tf.get_default_graph())
# </editor-fold>

# <editor-fold desc="2. 图的覆盖">
"""
利用上下文管理器(context manager) 进行图的覆盖.
感觉就是利用 with 在某张图中创建变量.
"""
print('==g=====')
g = tf.Graph() # 创建新图 g
print('g:', g)
with g.as_default(): # 设置 g as default
    d = tf.constant(value=2)
    print('d:', d.graph)
    print('def:', tf.get_default_graph())

print('==g2=====')
g2 = tf.Graph()
print('g2:', g2)
g2.as_default()
e = tf.constant(value=15) # e 不包含在 g 中, 因此是在最开始的图中.
print('e:', e.graph)
print('def:', tf.get_default_graph())
# </editor-fold>

# <editor-fold desc="3. 图的集合存储">
"""
collection 提供一个全局的存储机制, 不受到变量名生存空间的影响.
tf.add_to_collection: 
"""
print('==collections====')
tf.add_to_collection('v_c', c)
tf.add_to_collection('g_c', c.graph)
print(tf.get_default_graph().get_all_collection_keys())
print(tf.get_default_graph())
# </editor-fold>




print('end')
