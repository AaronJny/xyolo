# -*- coding: utf-8 -*-
# @Time    : 2020/10/31 15:22
# @Author  : AaronJny
# @File    : train.py
# @Desc    : 使用xyolo训练自己的模型
# 导入包
from xyolo import DefaultYolo3Config, YOLO
from xyolo import init_yolo_v3


# 创建一个DefaultYolo3Config的子类，在子类里覆盖默认的配置
class MyConfig(DefaultYolo3Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        # 数据集路径，推荐使用绝对路径
        self._dataset_path = '/home/aaron/tmp/test_xyolo/xyolo_data/yolo_label.txt'
        # 类别名称文件路径，推荐使用绝对路径
        self._classes_path = '/home/aaron/tmp/test_xyolo/xyolo_data/classes.txt'
        # 模型保存路径，默认是保存在当前路径下的xyolo_data下的，也可以进行更改
        # 推荐使用绝对路径
        self._output_model_path = '/home/aaron/tmp/test_xyolo/output_model.h5'


# 使用修改后的配置创建yolo对象
config = MyConfig()
init_yolo_v3(config)
# 如果是训练，在创建yolo对象时要传递参数train=True
yolo = YOLO(config, train=True)
# 开始训练，训练完成后会自动保存
yolo.fit()
