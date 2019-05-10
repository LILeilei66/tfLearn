## feature_column (特征列)

ref: https://www.tensorflow.org/guide/feature_columns#input_to_a_deep_neural_network

没看懂.

### 定义

feature_column 是原始数据和 Estimator 之间的媒介.

![1557404333027](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1557404333027.png)

### 九大函数

#### 1. numeric_column

```python
tf.feature_column.numeric_column(key, dtype, shape)
```

#### 2. bucketized_column()

完成的任务：每个桶代表一个label.

```python
numeric_feature_column = tf.feature_column.numeric_column('key')
tf.feature_column.bucketized_column(
	source_column = numeric_feature_column, 
	boundaries = [num1, num2, ...]
	)
```

#### 3. categorical_column_with_identity

作用: 类似于 bucketized_column(), 然而 bucketized 是通过 boundary value 来确定, 在 categorical_column_with_identity 中,  每个数就是一个label. 

```
identity_feature_column = tf.feature_column.categorical_column_with_identity(key='fta', num_bucket=4)

def input_fn():
	return ({'fta': [1,2,3,0], 'ftb': [5,6,7,8]},
			[label_values])
# fta in range(4)
```

### 4. categorical_column_with_vocabulary_list,   categorical_column_with_vocabulary_file

作用: 将字符串投影到 int.

```
vocabulary_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
	key=feature_name_from_input_fn,
	vocabulary_list=['fta', 'ftb', 'ftc']
	)

vocabulary_feature_column = tf.feature_column.categorical_column_with_vocabulay_file(
	key=feature_name_from_input_fn,
	vocabulary_file='product_class.txt',
	vocabulary_size=3
)
```

In product_class.txt:

```
fta
ftb
ftc
```

#### 5. categorical_column_with_hash_bucket

解决的问题: 当 class 的数量很多, 以致于无法为每个词汇或整数设置单独的类别.

作用:  **不懂。。。**

```
伪代码:
	feature_id = hash(raw_feature % hash_buckets_size)
hashed_feature_column = tf.feature_column.categorical_column_with_hash_bucket(
	key='some_feature',
	hash_buckets_size-100 # category number
)
```

#### 6. crossed_column

使用场景: features 有 fta, ftb. 分别看他们二者, 对于classification 并没有很大的影响, 但是如果将他们合成一个 feature_componented, 会是一个比较有效的feature.

```python
def make_dataset(fta, ftb, labels)
	fts = {'fta': fta.flatten(),
		   'ftb': ftb.flatten()
	}
	labels = labels.flatten()
	
	return tf.data.Dataset.from_tensor_slices((fts, labels))

# todo: input_fn 呢?
fta_bucket_fc = tf.feature_column.bucketized_column(
				tf.feature_column.numeric_column('fta'),
				list(datasource_fta)
	)
ftb_bucket_fc = tf.feature_column.bucketized_column(
				tf.feature_column.numeric_column('ftb'),
				list(datasource_ftb)
	)
cross_fc = tf.feature_column.crossed_column(
	[fta_bucket_fc, ftb_bucket_fc], 5000
)

fc = [fta_bucket_fc, ftb_bucket_fc, cross_fc]

est = tf.estimator.LinearRegressor(fc,...)
```