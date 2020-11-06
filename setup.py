# -*- coding: utf-8 -*-
# @Time    : 2020/10/30 23:54
# @Author  : AaronJny
# @File    : setup.py
# @Desc    :
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xyolo",  # Replace with your own username
    version="0.1.3",
    author="AaronJny",
    author_email="aaronjny7@gmail.com",
    description="A tf.keras implementation of YOLOv3 with TensorFlow 2.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AaronJny/tf2-keras-yolo3",
    packages=setuptools.find_packages(),
    package_data={
        'xyolo': ['xyolo_data/*.txt',
                  'xyolo_data/*.cfg']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'tensorflow>=2.2',
        'numpy>=1.18.1,<1.19.0',
        'pillow>=7.0.0',
        'matplotlib>=3.1.3',
        'loguru>=0.5.1',
        'requests>=2.22.0',
        'tqdm>=4.42.1',
        'lxml>=4.5.0',
        'opencv-python>=4.2.0'
    ]
)
