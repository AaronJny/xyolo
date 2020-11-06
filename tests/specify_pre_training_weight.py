# -*- coding: utf-8 -*-
# @Time    : 2020/11/6 22:03
# @Author  : AaronJny
# @File    : specify_pre_training_weight.py
# @Desc    : 指定预训练权重路径
from xyolo import YOLO, DefaultYolo3Config
from xyolo import init_yolo_v3


# 创建一个DefaultYolo3Config的子类，在子类里覆盖默认的配置
class MyConfig(DefaultYolo3Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        # 这是替换成你的文件路径，为了避免出错，请尽量使用绝对路径
        self._pre_training_weights_darknet_path = '/Users/aaron/data/darknet_yolo.weights'


# 使用修改后的配置创建yolo对象
config = MyConfig()
init_yolo_v3(config)
yolo = YOLO(config)

# 检测
img = yolo.detect_and_draw_image('./xyolo_data/detect.jpg')
img.show()
