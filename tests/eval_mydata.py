# -*- coding: utf-8 -*-
# @Time    : 2020/10/31 19:49
# @Author  : AaronJny
# @File    : eval_mydata.py
# @Desc    :
from xyolo import DefaultYolo3Config
from xyolo import YOLO


class MyConfig(DefaultYolo3Config):

    def __init__(self):
        super(MyConfig, self).__init__()
        self._classes_path = '/Users/aaron/code/xyolo/tests/xyolo_data/classes.txt'
        self._model_path = '/Users/aaron/code/xyolo/tests/xyolo_data/output_model.h5'


config = MyConfig()
yolo = YOLO(config)
image_path = '/Users/aaron/code/bctt/spider/captcha_detection/soopat/images/232.png'
img = yolo.detect_and_draw_image(image_path)
img.show()
