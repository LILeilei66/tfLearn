"""
官方doc:
    使用 tf.Session 进行 python 与 C++ 运行时的连接, 以及分布式 tf 计算.
    通过 tf.Session("grpc://example.org:2222") 进行远程访问.
功能:
    1. 进行 op 执行 <Operation.run()>;
    2. Tensor.eval()
参数:
    1. target <与分布式有关>;
    2. graph;
    3. config <cpu||gpu 配置>.


ref: 1. https://blog.csdn.net/xierhacker/article/details/53860379
     2. https://blog.csdn.net/u014281392/article/details/73878271
"""
import tensorflow as tf

# <editor-fold desc="1. 初等应用">
a = tf.Variable(initial_value=[[2.]])
b = tf.Variable(initial_value=[[5.]])

c = tf.matmul(a, b)
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    c = sess.run(c)

print(c) # [[10.]]
del a, b, c
# </editor-fold>

# <editor-fold desc="2. 参数 graph">
g = tf.Graph()

with tf.Session(graph=g) as sess:
    a = tf.Variable(initial_value=[2.0])
    b = tf.Variable(initial_value=[3.0])
    c = tf.multiply(a, b)
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    c = sess.run(c) # error
    print('c', c) # [6.]
# </editor-fold>

# <editor-fold desc="3. sess(fetch)">
"""支持 list, tuple, dict, named_tuple"""
a = tf.constant([10])
b= tf.constant([2])
with tf.Session() as sess:
    v1 = sess.run(a)
    print('v1', v1) # v1 [10]

    v_list = sess.run([a, b])
    print('v_list', v_list) # v_list [array([10]), array([2])]

    v_dict = sess.run({'a': a, 'b': b})
    print('v_dict', v_dict) # v_dict {'a': array([10]), 'b': array([2])}


    v_tuple = sess.run((a, b))
    print('v_tuple', v_tuple) # v_tuple (array([10]), array([2]))

# </editor-fold>

# <editor-fold desc="4. Tensor.eval()">
"""
===diff Session.run() Vs Tensor.eval()=====
stackoverflow说:
    Tensor t;
    t.eval() == tf.get_default_session().run(t)

ref: https://stackoverflow.com/questions/33610685/in-tensorflow-what-is-the-difference-between-session-run-and-tensor-eval
"""
a = tf.constant(100)
b = tf.constant(3)
c = tf.multiply(a, b)
d = tf.divide(a, b)

with tf.Session() as sess:
    print(c.eval()) # 300
    print(d.eval()) # 33.33333333333
# </editor-fold>