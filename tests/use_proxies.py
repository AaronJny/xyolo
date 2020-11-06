# -*- coding: utf-8 -*-
# @Time    : 2020/10/31 15:20
# @Author  : AaronJny
# @File    : use_proxies.py
# @Desc    : 使用预训练权重对图片执行目标检测
from xyolo import YOLO, DefaultYolo3Config
from xyolo import init_yolo_v3


# 创建一个DefaultYolo3Config的子类，在子类里覆盖默认的配置
class MyConfig(DefaultYolo3Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        # 这是替换成你的代理地址
        self.requests_proxies = {'https': 'http://localhost:7890'}


# 使用修改后的配置创建yolo对象
config = MyConfig()
init_yolo_v3(config)
yolo = YOLO(config)

# 检测
img = yolo.detect_and_draw_image('./xyolo_data/detect.jpg')
img.show()