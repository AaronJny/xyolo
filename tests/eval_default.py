# -*- coding: utf-8 -*-
# @Time    : 2020/10/31 15:20
# @Author  : AaronJny
# @File    : use_proxies.py
# @Desc    : 使用预训练权重对图片执行目标检测
# 导入包
# 导入包
from xyolo import YOLO, DefaultYolo3Config
from xyolo import init_yolo_v3


# 创建默认配置类对象
config = DefaultYolo3Config()
# 初始化xyolo（下载预训练权重、转换权重等操作都是在这里完成的）
# 下载和转换只在第一次调用的时候进行，之后再调用会使用缓存的文件
init_yolo_v3(config)
# 创建一个yolo对象，这个对象提供使用yolov3进行检测和训练的接口
yolo = YOLO(config)

# 检测并在图片上标注出物体
img = yolo.detect_and_draw_image('./xyolo_data/detect.jpg')
# 展示标注后图片
img.show()