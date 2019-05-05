"""
when shuffle's buffer_size is 1, no shuffle.
shuffle algorithm:
-----------------
1. take N samples from original dataset, N = buffer size
2. select one to he batch

repeat:
-------
据 csdn 说, tf 的 doc 表示, 使用 repeat before shuffle can improve the performance, well, I didn't find
this evaluation in the official doc.
algo: repeat the original dataset for count times and regard it as a whole dataset.
"""
import tensorflow as tf
tf.enable_eager_execution()

ds_tensor0 = tf.data.Dataset.from_tensor_slices([1,2,3,4])
ds_tensor1 = ds_tensor0.shuffle(3)
ds_tensor2 = ds_tensor0.batch(2)
ds_tensor3 = ds_tensor0.shuffle(3).batch(2)
ds_tensor4 = ds_tensor0.repeat(2)

root = 'ds_tensor'
for i in range(5):
    name = root + str(i)
    print(name)
    for tensor in eval(name):
        print(tensor)