import os
import math
import config
import tensorflow as tf
from models.ResNet import ResNetModel
from models.MobileNet import MobileNetV2
from dataset import NSFWDataset

# 指定使用GPU的Index
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index
tf.logging.set_verbosity(tf.logging.INFO)

def learning_rate_with_decay(base_lr, boundary_epochs, num_images, batch_size, decay_rates, global_step):
    """
    Introduction
    ------------
        学习率衰减函数
    Parameters
    ----------
        base_lr: 初始学习率
        boundary_epochs: 学习率衰减边界
        num_images: 数据集图片数量
        batch_size: 每个batch的图片数量
        decay_rates: 每次衰减的系数
        global_step: 全局步数
    """
    batches_per_epoch = num_images / batch_size
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [base_lr * decay for decay in decay_rates]
    lr = tf.train.piecewise_constant(global_step, boundaries, vals)
    return lr


def Resnet_model_fn(features, labels, mode):
    """
    Introduction
    ------------
        定义estimator输入模型函数
    Parameters
    ----------
        features: 输入的图片信息
        labels: 输入的图片标签
        mode: 标识训练、预测、验证
    """
    tf.summary.image('images', features, max_outputs = 6)
    # model = ResNetModel(config.num_classes, data_format = "channels_first")
    model = MobileNetV2(config.num_classes, data_format = "channels_first")
    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)
    predictions = {
        'classes': tf.argmax(logits, axis = 1),
        'probabilities': tf.nn.softmax(logits, name = 'softmax_tensor')
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions,
            export_outputs = {
                'predict': tf.estimator.export.PredictOutput(predictions)
            })
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits = logits, labels = labels)
    tf.summary.scalar("CrossEntropy", cross_entropy)
    vars = tf.trainable_variables()
    l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * config.weight_decay
    loss = cross_entropy + l2_loss
    tf.summary.scalar("loss", loss)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = learning_rate_with_decay(config.learning_rate, boundary_epochs=[30, 60, 80, 90], num_images = config.train_images,
        batch_size = config.train_batch_size, decay_rates = config.decay_rates, global_step = global_step)
        tf.summary.scalar("learning_rate", learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = config.momentum)
        def _dense_grad_filter(gvs):
            """
            Introduction
            ------------
                如果是进行fine tune, 需要过滤出最后一层的参数进行训练
            """
            return [(g, v) for g, v in gvs if 'dense' in v.name]
        grad_vars = optimizer.compute_gradients(loss)
        if config.fine_tune:
            grad_vars = _dense_grad_filter(grad_vars)
        minimize_op = optimizer.apply_gradients(grad_vars, global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None
    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    training_hook = tf.train.LoggingTensorHook({'loss' : loss, 'accuracy' : accuracy[1]}, every_n_iter = config.log_frequency)
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('train_accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions, loss = loss, train_op = train_op, eval_metric_ops = metrics, training_hooks = [training_hook])


def input_function(mode, dataset_dir, batch_size, num_epochs):
    """
    Introduction
    ------------
        构建模型的输入函数
    Parameters
    ----------
        mode: 训练、验证模式
        dataset_dir: 数据集路径
        batch_size: batch大小
        num_epochs: epoch大小
    """
    dataset = NSFWDataset(dataset_dir, mode).process_record_dataset(batch_size, num_epochs)
    return dataset

def build_tensor_serving_input_receiver_fn(shape):
    """
    Introduction
    ------------
        导出模型pb格式，定义输入函数
    Parameters
    ----------
        shape: 输入变量shape
        batch_size: batch大小
    """

    def serving_input_receiver_fn():
        features = tf.placeholder(tf.float32, shape = [None] + shape, name = "input_tensor")
        return tf.estimator.export.TensorServingInputReceiver(features = features, receiver_tensors = features)
    return serving_input_receiver_fn


def run():
    classifier = tf.estimator.Estimator(model_fn = Resnet_model_fn, model_dir = config.model_dir)
    n_loops = math.ceil(config.train_epochs / config.epochs_between_eval)
    schedule = [config.epochs_between_eval for _ in range(n_loops)]
    schedule[-1] = config.train_epochs - sum(schedule[:-1])
    for cycle_index, num_train_epochs in enumerate(schedule):
        tf.logging.info("start cycle:{}/{}".format(cycle_index, n_loops))
        if num_train_epochs:
            classifier.train(input_fn = lambda : input_function("train", config.data_dir, config.train_batch_size, num_train_epochs))
        tf.logging.info("start evaluate")
        eval_results = classifier.evaluate(input_fn = lambda : input_function("eval", config.data_dir, 1, 1))
        print(eval_results)
    input_receiver_fn = build_tensor_serving_input_receiver_fn([config.image_size, config.image_size, 3])
    classifier.export_savedmodel(config.export_dir, input_receiver_fn)

if __name__ == "__main__":
    run()