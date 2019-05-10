## 比较 pytorch 与 tensorflow 的 layer 建立

### 1. 全连接层

#### a) Tensorflow

```python
keras.layers.Dense(units, activation, use_bias, input_shape=(int,)) 以及一些 regularizer 等. input shape 可以没有
output shape:(*,32); output shape: (*, 16)
```

#### b) Pytorch

```python
nn.Linear(input_channel, out_put_channel, bias)
activation 以 layer 的形式, 另外创建: e.g.: nn.Sigmoid()
```



### 2. 卷积层

#### a) Tensorflow

```python
keras.layers.Conv2D(filters, kernel_size, strides=(), padding='', dilation_rate=(), data_format) 以及一些 regularizer 和 initializer 等.

output channel = filters; keinel size: of type tuple or list；
data_format: ['channels_first' || 'channels_last']
```

#### b) Pytorch

```python
nn.Conv2d(input_channel, output_channels, kernel_size, stride, padding, dilation, bias)
```



### 3. pooling

#### a) Tensorflow

```python
keras.layers.MaxPool2D(pool_size, stride, padding, data_format)
```

#### b) Pytorch


```python
nn.Maxpool2d(kernel_size, stride, padding, dilation, retrun_indices)
其他: Avgpool2d 等, 多种 dimension 等.
```