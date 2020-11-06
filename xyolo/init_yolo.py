# -*- coding: utf-8 -*-
# @Time    : 2020/10/31 00:18
# @Author  : AaronJny
# @File    : init_yolo.py
# @Desc    :
import os

import requests
from loguru import logger
from tqdm import tqdm

from xyolo.config import DefaultYolo3Config
from xyolo.convert import convert
from hashlib import md5


def compute_hash_code(filepath):
    """
    读取并计算给定文件的md5 hash值
    """
    with open(filepath, 'rb') as f:
        data = f.read()
    return md5(data).hexdigest()


def download_weights(config):
    darknet_path = config.pre_training_weights_darknet_path
    if os.path.exists(darknet_path):
        # 如果已经存在，先校验md5哈希值
        current_hash_code = compute_hash_code(darknet_path)
        # md5相同才说明已经下载了，否则重新下载
        if current_hash_code == config.pre_training_weights_darknet_md5:
            logger.info('Pre-training weights already exists! Skip!')
            return
    weights_url = config.pre_training_weights_url
    r = requests.get(weights_url, stream=True, proxies=config.requests_proxies)
    filename = weights_url.split('/')[-1]
    with tqdm.wrapattr(open(darknet_path, "wb"), "write",
                       miniters=1, desc=filename,
                       total=int(r.headers.get('content-length', 0))) as f:
        for chunk in r.iter_content(chunk_size=1024 * 100):
            if chunk:
                f.write(chunk)
    logger.info('Saved Darknet model to {}'.format(darknet_path))


def init_yolo_v3(config=None):
    if not config:
        config = DefaultYolo3Config()
    logger.info('Downloading Pre-training weights of yolo v3 ...')
    download_weights(config)
    logger.info('Convert Darknet -> Keras ...')
    if os.path.exists(config.pre_training_weights_keras_path):
        logger.info('Keras model already exists! Skip!')
    else:
        convert(config_path=config.pre_training_weights_config_path,
                weights_path=config.pre_training_weights_darknet_path,
                output_path=config.pre_training_weights_keras_path)
    logger.info('Init completed.')


if __name__ == '__main__':
    init_yolo_v3()
