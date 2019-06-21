import time
import config
import argparse
import tensorflow as tf
from models.MobileNet import MobileNetV2

class predictor:
    def __init__(self, modelPath):
        """
        Introduction
        ------------
            模型预测模块初始化
        Parameters
        ----------
            modelPath: 模型路径
        """
        self.sess = tf.Session()
        self.modelPath = modelPath

    def _load_model(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, self.modelPath)

    def __call__(self, image_file, training = False):
        image_data = tf.gfile.FastGFile(image_file, 'rb').read()
        image = tf.image.decode_jpeg(image_data, channels = 3)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        image = tf.image.resize_images(image, [config.image_size, config.image_size])
        image = tf.image.per_image_standardization(image)
        image = tf.expand_dims(image, dim=0)
        startTime = time.time()

        model = MobileNetV2(config.num_classes, data_format = "channels_last")
        logits = model(image, training)
        classes = tf.argmax(logits, axis = 1)
        probabilities = tf.nn.softmax(logits)
        self._load_model()
        predictLabel, predictProb = self.sess.run([classes, probabilities])
        print("预测结果：{} 概率值：{}".format(config.class_dict[predictLabel[0]], predictProb[0]))
        print("预测耗时：{}".format(time.time() - startTime))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default = argparse.SUPPRESS)
    parser.add_argument(
        '--image_file', type = str, help = 'image file path'
    )
    FLAGS = parser.parse_args()
    predict = predictor("mobilenet_model/model.ckpt-1165151")
    predict(FLAGS.image_file, training = False)
