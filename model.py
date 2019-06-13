import config
import tensorflow as tf

def batch_norm(inputs, training, data_format):
    """
    Introduction
    ------------
        batch norm层
    Args
    ----
        inputs: 输入变量
        training: 是否训练
        data_format: 数据格式
    Returns
    -------
        计算结果
    """
    return tf.layers.batch_normalization(inputs, axis=1 if data_format == 'channels_first' else 3,
                                         momentum = config.momentum, epsilon = config.epsilon, center = True, scale = True, training = training, fused = True)


def fixed_padding(inputs, kernel_size, data_format):
    """
    Introduction
    ------------
        对图像特征进行padding
    Args
    ----
        inputs: 输入图像
        kernel_size: 卷积核大小
        data_format: 数据格式
    Returns
    -------
        计算结果
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(tensor = inputs, paddings = [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(tensor = inputs, paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def Conv2d_pading(inputs, filters, kernel_size, strides, data_format):
    """
    Introduction
    ------------
        对输入图像进行卷积操作
    Args
    ----
        inputs: 输入图像特征
        filters: 卷积核数量
        kernel_size: 卷积核大小
        strides: 卷积步长
        data_format: 输入数据格式
    Returns
    -------
        计算结果
    """
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.layers.conv2d(inputs, filters, kernel_size, strides, padding = ('SAME' if strides == 1 else 'VALID'),
                            data_format = data_format, use_bias = False, kernel_initializer = tf.variance_scaling_initializer())


def bottleneck(inputs, filters, training, strides, data_format, downSample = False):
    """
    Introduction
    ------------
        构建resnet的block
    Args
    ----
        inputs: 输入图像
        filters: 卷积核个数
        training: 是否训练
        strides: 卷积步长
        data_format: 数据格式
    Returns
    -------
        计算结果
    """
    #如果图像大小降采样，shortcut也需要做降采样
    shortcut = inputs
    if downSample == True:
        shortcut = Conv2d_pading(inputs = inputs, filters = filters * 4, kernel_size = 1, strides = strides, data_format = data_format)
        shortcut = batch_norm(inputs = shortcut, training = training, data_format = data_format)

    #三层卷积, 由窄到宽
    inputs = Conv2d_pading(inputs = inputs, filters = filters, kernel_size = 1, strides = 1, data_format = data_format)
    inputs = batch_norm(inputs = inputs, training = training, data_format = data_format)
    inputs = tf.nn.relu(inputs)
    inputs = Conv2d_pading(inputs = inputs, filters = filters, kernel_size = 3, strides = strides, data_format = data_format)
    inputs = batch_norm(inputs = inputs, training = training, data_format = data_format)
    inputs = tf.nn.relu(inputs)
    inputs = Conv2d_pading(inputs = inputs, filters = filters * 4, kernel_size = 1, strides = 1, data_format = data_format)
    inputs = batch_norm(inputs = inputs, training = training, data_format = data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def make_layer(inputs, filters, strides, blocks, name, training, data_format):
    """
    Introduction
    ------------
        构建resnet block层
    Args
    ----
        inputs: 输入变量
        filters: 卷积核数
        strides: 卷积步长
        blocks: block个数
        training: 是否为训练阶段
        name: block的名字
        data_format: 数据格式
    Returns
    -------
        每个block计算结果
    """
    inputs = bottleneck(inputs = inputs, filters = filters, training = training, strides = strides, data_format = data_format, downSample = True)
    for _ in range(1, blocks):
        inputs = bottleneck(inputs = inputs, filters = filters, training = training, strides = 1, data_format = data_format, downSample = False)
    return tf.identity(inputs, name)


class ResNetModel:
    def __init__(self, block_sizes, num_classes, data_format = None):
        """
        Introduction
        ------------
            模型初始化
        Args
        ----
            block_sizes: Resnet每个block层数List
            num_classes: 图片分类数目
            data_format: 输入数据格式, channels first or channels last
        """
        self.block_sizes = block_sizes
        self.num_classes = num_classes
        self.data_format = data_format

    def __call__(self, inputs, training):
        """
        Introduction
        ------------
            模型进行训练或者预测forward阶段
        Args
        ----
            inputs: 输入图像数据
            training: bool变量，标志是否为训练阶段
        Returns
        -------
            返回全连接层输出
        """
        with tf.variable_scope("resnet50"):
            if self.data_format == "channels_first":
                inputs = tf.transpose(inputs, perm = [0, 3, 1, 2])
            inputs = Conv2d_pading(inputs = inputs, filters = 64, kernel_size = 7, strides = 2, data_format = self.data_format)
            inputs = batch_norm(inputs = inputs, training = training, data_format = self.data_format)
            inputs = tf.nn.relu(inputs)
            inputs = tf.layers.max_pooling2d(inputs = inputs, pool_size = 3, strides = 2, padding = 'SAME', data_format = self.data_format)
            BlockStrides = [1, 2, 2, 2]
            for i, block_nums in enumerate(self.block_sizes):
                inputs = make_layer(inputs, filters = 64 * (2**i), strides = BlockStrides[i], blocks = block_nums,
                                    name = "block_layer{}".format(i+1), training = training, data_format = self.data_format)
            axis = [2, 3] if self.data_format == 'channels_first' else [1, 2]
            inputs = tf.reduce_mean(inputs, axis = axis)
            inputs = tf.identity(inputs, 'final_reduce_mean')

            inputs = tf.squeeze(inputs, axis = axis)
            inputs = tf.layers.dense(inputs = inputs, units = self.num_classes)
            inputs = tf.identity(inputs, 'final_dense')
            return inputs
