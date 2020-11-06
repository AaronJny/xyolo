# -*- coding: utf-8 -*-
# @Time    : 2020/10/31 10:50
# @Author  : AaronJny
# @File    : config.py
# @Desc    :
from os.path import abspath, join, dirname, exists
from os import mkdir


class DefaultYolo3Config:
    """
    yolo3模型的默认设置
    """

    def __init__(self):
        # xyolo各种数据的保存路径，包内置
        self.inner_xyolo_data_dir = abspath(join(dirname(__file__), './xyolo_data'))
        # xyolo各种数据的保存路径，包外，针对于项目
        self.outer_xyolo_data_dir = abspath('./xyolo_data')
        # yolo3预训练权重下载地址
        self.pre_training_weights_url = 'https://pjreddie.com/media/files/yolov3.weights'
        # 下载文件时的http代理，
        # 如需设置，格式为{'https_proxy':'host:port'}，如{'https_proxy':'http://127.0.0.1:7890'},
        # 详细设置请参考 https://requests.readthedocs.io/en/master/user/advanced/#proxies
        self.requests_proxies = None
        # Darknet格式的预训练权重路径,请填写相对于inner_xyolo_data_dir的相对或绝对路径
        self._pre_training_weights_darknet_path = 'darknet_yolo.weights'
        # yolo3预训练权重darknet md5 hash值，用于处理异常数据
        self.pre_training_weights_darknet_md5 = 'c84e5b99d0e52cd466ae710cadf6d84c'
        # 转化后的、Keras格式的预训练权重路径，请填写相对于inner_xyolo_data_dir的相对或绝对路径
        self._pre_training_weights_keras_path = 'keras_weights.h5'
        # 预训练权重的配置路径，请填写相对于inner_xyolo_data_dir的相对或绝对路径
        self._pre_training_weights_config_path = 'yolov3.cfg'
        # 默认的anchors box路径，请填写相对于inner_xyolo_data_dir的相对或绝对路径
        self._anchors_path = 'yolo_anchors.txt'
        # 默认的类别文本路径，请填写相对于inner_xyolo_data_dir的相对或绝对路径
        self._classes_path = 'coco_classes.txt'
        # 训练输出的模型地址,请填写相对于outer_xyolo_data_dir的相对或绝对路径
        self._output_model_path = 'output_model.h5'
        # 数据集路径,请填写相对于outer_xyolo_data_dir的相对或绝对路径
        self._dataset_path = 'dataset.txt'
        # 是否开启TensorBoard，默认开启
        self.use_tensorboard = True
        # 训练时TensorBoard输出路径，请填写相对于outer_xyolo_data_dir的相对或绝对路径
        self._tensorboard_log_path = './tensorboard/logs'
        # 是否开启CheckPoint，默认开启
        self.use_checkpoint = True
        # 是否开启学习率衰减
        self.use_reduce_lr = True
        # 学习率衰减监控指标，默认为验证loss
        self.reduce_lr_monitor = 'val_loss'
        # 学习率衰减因子，new_lr = lr * factor
        self.reduce_lr_factor = 0.1
        # 连续patience个epochs内结果未改善，则进行学习率衰减
        self.reduce_lr_patience = 3
        # 是否开启early_stopping
        self.use_early_stopping = True
        # early_stopping监控指标，默认为验证loss
        self.early_stopping_monitor = 'val_loss'
        # 指标至少变化多少认为结果改善了
        self.early_stopping_min_delta = 0
        # 连续patience个epochs内结果未改善，则提前结束训练
        self.early_stopping_patience = 10
        # yolo默认加载的模型路径(最好填写绝对路径)，优先级设置见下方model_path方法
        self._model_path = ''
        # 目标检测分数阈值
        self.score = 0.3
        # 交并比阈值
        self.iou = 0.45
        # 模型图片大小
        self.model_image_size = (416, 416)
        # GPU数量
        self.gpu_num = 1
        # 训练时的验证集分割比例，默认为0.1，
        # 即将数据集中90%的数据用于训练，10%的用于测试
        self.val_split = 0.1
        # 训练分为两步，第一步冻结大多数层进行训练，第二步解冻进行微调
        # 是否开启冻结训练，建议开启
        self.frozen_train = True
        # 冻结时，训练的epoch数
        self.frozen_train_epochs = 50
        # 冻结时，训练的batch_size
        self.frozen_batch_size = 32
        # 冻结时的初始学习率
        self.frozen_lr = 1e-3
        # 是否开启解冻训练，建议开启
        self.unfreeze_train = True
        # 解冻时，训练的epoch数
        self.unfreeze_train_epochs = 50
        # 解冻时，训练的batch_size.注意，解冻时训练对GPU内存需求量非常大，这里建议设置小一点
        self.unfreeze_batch_size = 1
        # 解冻时的初始学习率
        self.unfreeze_lr = 1e-4

    def __setattr__(self, key, value):
        _key = '_{}'.format(key)
        if key not in self.__dict__ and _key in self.__dict__:
            self.__dict__[_key] = value
        else:
            self.__dict__[key] = value

    @classmethod
    def make_dir(cls, path):
        if not exists(path):
            mkdir(path)

    @classmethod
    def join_and_abspath(cls, path1, path2):
        return abspath(join(path1, path2))

    def inner_abspath(self, filename):
        self.make_dir(self.inner_xyolo_data_dir)
        return self.join_and_abspath(self.inner_xyolo_data_dir, filename)

    def outer_abspath(self, filename):
        self.make_dir(self.outer_xyolo_data_dir)
        return self.join_and_abspath(self.outer_xyolo_data_dir, filename)

    @property
    def pre_training_weights_darknet_path(self):
        return self.inner_abspath(self._pre_training_weights_darknet_path)

    @property
    def pre_training_weights_config_path(self):
        return self.inner_abspath(self._pre_training_weights_config_path)

    @property
    def pre_training_weights_keras_path(self):
        return self.inner_abspath(self._pre_training_weights_keras_path)

    @property
    def anchors_path(self):
        return self.inner_abspath(self._anchors_path)

    @property
    def classes_path(self):
        return self.inner_abspath(self._classes_path)

    @property
    def output_model_path(self):
        return self.outer_abspath(self._output_model_path)

    @property
    def dataset_path(self):
        return self.outer_abspath(self._dataset_path)

    @property
    def tensorboard_log_path(self):
        return self.outer_abspath(self._tensorboard_log_path)

    @property
    def model_path(self):
        """
        Yolo模型默认加载的权重的路径。
        按照 _model_path > output_model_path > pre_training_weights_keras_path 的优先级选择，即：
        如果设置了_model_path,选择_model_path
        否则，如果设置了output_model_path且路径存在，选择output_model_path
        否则，选择pre_training_weights_keras_path
        """
        _model_path = getattr(self, '_model_path', '')
        if _model_path:
            return abspath(_model_path)
        if self._output_model_path and exists(self.output_model_path):
            return self.output_model_path
        return self.pre_training_weights_keras_path
