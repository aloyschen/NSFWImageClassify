# encoding:utf-8
import os
import math
import random
import config
import tensorflow as tf

class NSFWDataset():
    def __init__(self, datasetDir, mode):
        """
        Introduction
        ------------
            图像数据集
            1、将图像数据转换为tfRecord
        """
        self.datasetDir = datasetDir
        self.mode = mode
        self._sess = tf.Session()
        file_pattern = os.path.join(self.datasetDir, self.mode) + '/tfrecords/*.tfrecord'
        self.tfRecord_file = tf.gfile.Glob(file_pattern)
        self._encode_image = tf.placeholder(tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encode_image, channels = 3)
        self._decode_png = tf.image.decode_png(self._encode_image, channels = 3)
        if len(self.tfRecord_file) == 0:
            self.convert_to_tfecord()
            self.tfRecord_file = tf.gfile.Glob(file_pattern)

    def int64_feature(self, values):
        """
        Introduction
        ------------
            转换成tensorflow tfrecord int特征格式
        """
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(int64_list = tf.train.Int64List(value = values))


    def bytes_feature(self, values):
        """
        Introduction
        ------------
            转换成tensorflow tfrecord bytes特征格式
        """
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [values]))


    def _get_filenames_and_classes(self):
        """
        Introduction
        ------------
            获取路径下对应的图片和所有的类别
        Parameters
        ----------
            dataset_dir: 数据集对应的路径
            mode: 数据集对应的训练、测试、验证
        Returns
        -------
            返回数据集包含的所有图片路径和所有的类别名称
        """
        image_path = []
        classes_name = []
        root_path = os.path.join(self.datasetDir, self.mode)
        for filename in os.listdir(root_path):
            path = os.path.join(root_path, filename)
            if os.path.isdir(path):
                classes_name.append(filename)
                for imageFile in os.listdir(path):
                    image_path.append(os.path.join(path, imageFile))
        return image_path, sorted(classes_name)


    def PreProcessImage(self, image):
        """
        Introduction
        ------------
            对图片进行预处理
        Parameters
        ----------
            image: 输入图片
        Returns
        -------
            预处理之后的图片
        """
        if self.mode == 'train':
            image = tf.image.resize_image_with_crop_or_pad(image, config.image_size, config.image_size)
            image = tf.image.random_flip_left_right(image)
        # 对图片像素进行标准化，减去均值，除以方差
        image = tf.image.per_image_standardization(image)
        return image


    def convert_to_tfecord(self):
        """
        Introduction
        ------------
            将数据集转换为tfrecord格式
        Parameters
        ----------
        """
        image_files, classes = self._get_filenames_and_classes()
        random.seed(0)
        random.shuffle(image_files)
        class_id_dict = dict(zip(classes, range(len(classes))))
        if self.mode == "train":
            num_shards = 200
        else:
            num_shards = 10
        num_per_shard = int(math.ceil(len(image_files) / float(num_shards)))
        image_nums = 0
        for shard_id in range(num_shards):
            output_filename = os.path.join(self.datasetDir, self.mode) + "/tfrecords/nsfw_{}_{}_of_{}.tfrecord".format(self.mode, shard_id, num_shards)
            with tf.python_io.TFRecordWriter(output_filename) as tfRecordWriter:
                start_idx = shard_id * num_per_shard
                end_idx = min((shard_id + 1) * num_per_shard, len(image_files))
                for idx in range(start_idx, end_idx):
                    print("converting image {}/{} shard {}".format(idx, len(image_files), shard_id))
                    image_data = tf.gfile.FastGFile(image_files[idx], 'rb').read()
                    # 数据可能有问题，若抛出异常则舍弃这条数据
                    try:
                        if image_files[idx].split('.')[-1] == 'png':
                            image = self._sess.run(self._decode_png, feed_dict = {self._encode_image : image_data})
                            assert len(image.shape) == 3
                            assert image.shape[2] == 3
                        else:
                            image = self._sess.run(self._decode_jpeg, feed_dict = {self._encode_image : image_data})
                            assert len(image.shape) == 3
                            assert image.shape[2] == 3
                    except Exception:
                        continue
                    height, width = image.shape[0], image.shape[1]
                    classname = os.path.basename(os.path.dirname(image_files[idx]))
                    class_id = class_id_dict[classname]
                    example = tf.train.Example(features = tf.train.Features(feature ={
                        'image/encoded' : self.bytes_feature(image_data),
                        'image/label' : self.int64_feature(class_id),
                        'image/height' : self.int64_feature(height),
                        'image/width' : self.int64_feature(width)
                    }))
                    tfRecordWriter.write(example.SerializeToString())
                    image_nums += 1
        print("所有数据集数量", image_nums)


    def Parse(self, serialized_example):
        """
        Introduction
        ------------
            解析tfrecord文件
        Parameters
        ----------
            serialized_example: 序列化数据
        """
        parsed = tf.parse_single_example(
            serialized_example,
            features = {
                'image/encoded' : tf.FixedLenFeature([], tf.string),
                'image/label' : tf.FixedLenFeature([], tf.int64),
                'image/height' : tf.FixedLenFeature([], tf.int64),
                'image/width' : tf.FixedLenFeature([], tf.int64)
            })
        image = tf.image.decode_jpeg(parsed['image/encoded'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        image.set_shape([None, None, 3])
        image = self.PreProcessImage(image)
        if self.mode != 'train':
            image = tf.image.resize_images(image, [config.image_size, config.image_size])
        label = parsed['image/label']
        label = tf.cast(label, tf.int32)
        return image, label

    def process_record_dataset(self, batch_size, num_epochs):
        """
        Introduction
        ------------
            返回tensorflow 训练的dataset
        Parameters
        ----------
            batch_size: 数据集每个batch的大小
            num_epochs: 数据集训练的轮数
        """
        dataset = tf.data.TFRecordDataset(filenames = self.tfRecord_file)
        dataset = dataset.map(self.Parse, num_parallel_calls = 1)
        dataset = dataset.batch(batch_size).prefetch(buffer_size = batch_size)
        if self.mode == 'train':
            dataset = dataset.shuffle(buffer_size = 500)
        dataset = dataset.repeat(num_epochs)

        return dataset

