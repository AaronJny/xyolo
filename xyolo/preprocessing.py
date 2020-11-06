# -*- coding: utf-8 -*-
# @Time    : 2020/11/3 14:26
# @Author  : AaronJny
# @File    : preprocessing.py
# @Desc    :
import xml.etree.ElementTree as ET
from glob import glob

from tqdm import tqdm


def _voc2xyolo(xml_path, classes):
    in_file = open(xml_path)
    tree = ET.parse(in_file)
    root = tree.getroot()
    image_path = root.find('path').text
    ret = [image_path, ]
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes[cls]
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        ret.append(",".join([str(a) for a in b]) + ',' + str(cls_id))
    return ' '.join(ret)


def voc2xyolo(input_path, classes_path, output_path):
    """
    将voc格式的标注数据转换成xyolo接受的类型

    Args:
        input_path: 输入文件路径的正则表达式。这里是使用labelImg标注的图片label文件路径
        classes_path: 保存实体类别的文件路径
        output_path: 转换后的数据集保存路径
    """
    with open(classes_path, 'r', encoding='utf8') as f:
        lines = [line.strip() for line in f.readlines()]
    classes = dict(zip(lines, range(len(lines))))
    files = glob(input_path)
    xyolo_lines = []
    for xml_path in tqdm(files):
        xyolo_line = _voc2xyolo(xml_path, classes)
        xyolo_lines.append(xyolo_line)
    with open(output_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(xyolo_lines))
