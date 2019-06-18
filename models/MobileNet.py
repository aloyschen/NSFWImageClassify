import config
import tensorflow as tf

def batch_norm(inputs, training, data_format):
    """
    Introduction
    ------------
        batch norm层
    Parameters
    ----------
        inputs: 输入变量
        training: 是否训练
        data_format: 数据格式
    Returns
    -------
        计算结果
    """
    return tf.layers.batch_normalization(inputs, axis=1 if data_format == 'channels_first' else 3,
            momentum = config.batchNorm_momentum, epsilon = config.batchNorm_epsilon, center = True, scale = True, training = training, fused = True)


def InvertedResidual(inputs, expand_ratio, channels, stride, training, data_format = 'channels_last'):
    """
    Introduction
    ------------
        定义MobileNet模块结构
    Parameters
    ----------
        inputs: 输入特征
        expand_ratio: 扩充比例
        channels: 输出通道数
        stride: 步长
        training: 是否训练
        data_format: 数据格式
    """
    if data_format == 'channels_first':
        filters = expand_ratio * inputs.get_shape().as_list[1]
    else:
        filters = expand_ratio * inputs.get_shape().as_list[-1]
    output = inputs
    if expand_ratio != 1:
        # pw
        output = tf.layers.conv2d(output, filters, kernel_size = 1, strides = 1, padding = 'VALID', use_bias = False,
                                  data_forma = data_format, kernel_initializer = tf.glorot_uniform_initializer())
        output = batch_norm(output, training, data_format)
        output = tf.nn.relu6(output)
    # dw
    output = tf.layers.separable_conv2d(output, None, kernel_size = 3, strides = stride, padding = 'SAME', use_bias = False,
                               data_format = data_format, depthwise_initializer = tf.glorot_uniform_initializer())
    output = batch_norm(output, training, data_format)
    output = tf.nn.relu6(output)

    # pw-linear
    output = tf.layers.conv2d(output, channels, kernel_size = 1, strides = 1, padding = 'VALID', use_bias = False,
                              data_format = data_format, kernel_initializer = tf.glorot_uniform_initializer())
    output = batch_norm(output, training, data_format)
    if stride == 1 and filters == channels:
        output = tf.add(inputs, output)
    return output


class MobileNetV2:
    def __init__(self, num_classes, data_format = "channels_last"):
        """
        Introduction
        ------------
            MobileNetV2模型初始化
        """
        self.data_format = data_format
        self.num_classes = num_classes
        self.last_channels = 1280
        self.inverted_residual_config = [
            # t (expand ratio), channel, n (layers), stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]


    def __call__(self, input, training):
        with tf.variable_scope("MobilenetV2"):
            input = tf.layers.conv2d(input, filters = 32, kernel_size = 3, strides = 2, padding = "SAME", use_bias = False,
                                     data_format = self.data_format, kernel_initializer = tf.glorot_uniform_initializer())
            input = batch_norm(input, training)
            input = tf.nn.relu6(input)
            for t, c, n, s in self.inverted_residual_config:
                for j in range(n):
                    if j == 0:
                        input = InvertedResidual(input, expand_ratio = t, channels = c, stride = s,
                                                 training = training, data_format = self.data_format)
                    else:
                        input = InvertedResidual(input, expand_ratio = t, channels = c, stride = 1,
                                                 training = training, data_format = self.data_format)

            input = tf.layers.conv2d(input, filters = self.last_channels, kernel_size = 1, strides = 1, padding = "VALID", use_bias = False,
                                     data_format = self.data_format, kernel_initializer = tf.glorot_uniform_initializer())
            input = batch_norm(input, training)
            input = tf.nn.relu6(input)

            axis = [2, 3] if self.data_format == 'channels_first' else [1, 2]
            input = tf.reduce_mean(input, axis = axis)
            input = tf.identity(input, 'final_reduce_mean')

            input = tf.reshape(input, [-1, self.last_channels])
            input = tf.layers.dense(inputs = input, units = self.num_classes)
            input = tf.identity(input, 'final_dense')

            return input