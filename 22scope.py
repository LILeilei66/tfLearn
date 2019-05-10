"""
有时会碰到 argument name = scope, 但是这究竟是什么呢？
scope 是一种给变量命名的方法, 具体有两种用法:
    1. name_scope
    2. variable_scope
ref:
1. https://blog.csdn.net/u012436149/article/details/53081454
2. https://blog.csdn.net/WIinter_FDd/article/details/70215844
"""
import tensorflow as tf

# <editor-fold desc="0. 作用整理">
"""
鉴于 tf 中节点数量极多, 使用 op/tensor 的方式进行命名.
tf.name_scope('0.1') + tf.Variable() -> scope_name of '0.1'
tf.variable_scope('0.2') + tf.get_variable() -> scope_name of 0.2

Rq:
name_scope 总能给 op 添加 scope_name
"""

# </editor-fold>

# <editor-fold desc="1. name scope">
"""name_scope 本质上是一个 string, 并没有给 variable 造成什么"""
print('1.')
with tf.name_scope("1.") as name_scope:
    a = tf.get_variable('a', shape=[2,2], dtype=tf.float32)

    print('name_scope:{:}, type:{:}. '.format(name_scope, type(name_scope)))
    # name_scope:1./, type:<class 'str'>.
    print('a.name:', a.name) # a.name: a:0
    print('original_name_scope:', tf.get_variable_scope().original_name_scope)
    # original_name_scope:

del a, name_scope
# </editor-fold>

# <editor-fold desc="2. variable scope">
"""variable_scope 得到的是一个 VariableScope object"""
print('2.')
with tf.variable_scope('2.') as variable_scope:
    a = tf.get_variable('a', shape=[2,10], dtype=tf.float32)

    print('variable_scope:{:}, \nname:{:}, type:{:}. '.format( \
        variable_scope, variable_scope.name, type(variable_scope)))
    # variable_scope:<tensorflow.python.ops.variable_scope.VariableScope object at
    # 0x00000204529BDBA8>,
    # name:2., type:<class 'tensorflow.python.ops.variable_scope.VariableScope'>.

    print('a:', a.name)
    # a: 2. / a:0

    print('original_name_scope', tf.get_variable_scope().original_name_scope)
    # original_name_scope 2./

    with tf.variable_scope('2.1') as scope2:
        print(tf.get_variable_scope().original_name_scope)
        # 2./2.1/

    with tf.name_scope('2.2') as scope3:
        print(tf.get_variable_scope().original_name_scope)
        # 2./
del a, variable_scope, scope2
# </editor-fold>

# <editor-fold desc="3. name_scope variable_scope with operation">
"""variable_scope 与 name_scope 都会给 op 的 name 加上前缀."""
print('3.')
with tf.name_scope('3.'):
    with tf.variable_scope('3.1'):
        w = tf.get_variable('w', shape=[2,2])
        result = tf.multiply(w, 3)
print('w.name:{:}, type:{:}'.format(w.name, type(w)))
# w.name:3.1/w:0, type:<class 'tensorflow.python.ops.variables.RefVariable'>
print('result.name:{:}, type:{:}'.format(result.name, type(result)))
# result.name:3./3.1/Mul:0, type:<class 'tensorflow.python.framework.ops.Tensor'>
del w, result
# </editor-fold>

# <editor-fold desc="4. 串联 scopes">
print('4.')
with tf.name_scope('4.1.1') as scope1:
    with tf.name_scope('4.1.2') as scope2:
        print(scope2)
# 4.1.1/4.1.2/

with tf.variable_scope('4.2.1') as scope1:
    with tf.variable_scope('4.2.2') as scope2:
        print('scope2:{:}\nscope2.name:{:}'.format( \
            scope2, scope2.name))
# scope2:<tensorflow.python.ops.variable_scope.VariableScope object at 0x000001693D251278>
# scope2.name:4.2.1/4.2.2
del scope1, scope2
# </editor-fold>

# <editor-fold desc="5. name_scope + Variable">
"""name_scope 可以为 Variable 添加 scope"""
print('5.')
with tf.name_scope('5.') as scope:
    a = tf.constant(5, name='5.1')
    b = tf.Variable(initial_value=[2], name='5.2')
    c = tf.Variable(initial_value=[3], name='5.3')
print(a.name)
# 5./5.1:0
print(b.name)
# 5./5.2:0
print(c.name)
# 5./5.3:0
del a, b, c
# </editor-fold>

# <editor-fold desc="6. variable_scope + Variable">
print('6.')
with tf.variable_scope('6.') as scope:
    a = tf.Variable(initial_value=[2], name='6.1')
print(a.name)
# 6./6.1:0
del a
# </editor-fold>

# <editor-fold desc="7. variable_scope + conflict">
"""当有两个不应是同一个 scope 的 variable 却定义成同一个 scope 时, 会 + '_1'"""
print('7.')
def test(name=None):
    with tf.variable_scope(name, default_name='7.1') as scope:
        a = tf.get_variable('7.1.1', shape=[2])
test()
test()
ws = tf.global_variables()
for w in ws:
    print(w.name)
# ...
# 7.1/7.1.1:0
# 7.1_1/7.1.1:0
del ws, test
# </editor-fold>

# <editor-fold desc="8. del name scope by name_scope">
print('8.')
with tf.name_scope('8.1'):
    a = tf.Variable(initial_value=[1], name='8.1.1')
    with tf.name_scope(None) as scope:
        b = tf.Variable(initial_value=[1], name='8.2')
print(a.name)
# 8.1/8.1.1:0
print(b.name)
# 8.2:0
del a, b
# </editor-fold>