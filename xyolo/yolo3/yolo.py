# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
import typing
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from PIL import Image
from loguru import logger
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

from xyolo.config import DefaultYolo3Config
from xyolo.yolo3.model import create_model, create_tiny_model
from xyolo.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body, preprocess_true_boxes
from xyolo.yolo3.utils import get_random_data
from xyolo.yolo3.utils import letterbox_image


class YOLO(object):

    def __init__(self, config=None, train=False, **kwargs):
        if not config:
            config = DefaultYolo3Config()
        self.config = config
        self.model_path = config.model_path
        self.anchors_path = config.anchors_path
        self.classes_path = config.classes_path
        self.score = config.score
        self.iou = config.iou
        self.model_image_size = config.model_image_size
        self.gpu_num = config.gpu_num
        self.dataset_path = config.dataset_path
        self.__dict__.update(kwargs)  # update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        if not train:
            self.load_yolo_model()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_yolo_model(self):
        self.model_path = self.config.model_path
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

    @tf.function
    def compute_output(self, image_data, image_shape):
        # Generate output tensor targets for filtered bounding boxes.
        # self.input_image_shape = K.placeholder(shape=(2,))
        self.input_image_shape = tf.constant(image_shape)
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)

        boxes, scores, classes = yolo_eval(self.yolo_model(image_data), self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    @classmethod
    def data_generator(cls, annotation_lines, batch_size, input_shape, anchors, num_classes):
        '''data generator for fit_generator'''
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(batch_size):
                if i == 0:
                    np.random.shuffle(annotation_lines)
                image, box = get_random_data(annotation_lines[i], input_shape, random=True)
                image_data.append(image)
                box_data.append(box)
                i = (i + 1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(batch_size)

    @classmethod
    def data_generator_wrapper(cls, annotation_lines, batch_size, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        if n == 0 or batch_size <= 0: return None
        return cls.data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

    def fit(self, **kwargs):
        # 如下参数可以通过传参覆盖config里的配置，更灵活一些
        # 不配置默认使用config里的参数
        dataset_path = kwargs.get('dataset_path', self.config.dataset_path)
        tensorboard_log_path = kwargs.get('tensorboard_log_path', self.config.tensorboard_log_path)
        output_model_path = kwargs.get('output_model_path', self.config.output_model_path)
        frozen_train = kwargs.get('frozen_train', self.config.frozen_train)
        frozen_train_epochs = kwargs.get('frozen_train_epochs', self.config.frozen_train_epochs)
        frozen_batch_size = kwargs.get('frozen_batch_size', self.config.frozen_batch_size)
        frozen_lr = kwargs.get('frozen_lr', self.config.frozen_lr)
        unfreeze_train = kwargs.get('unfreeze_train', self.config.unfreeze_train)
        unfreeze_train_epochs = kwargs.get('unfreeze_train_epochs', self.config.unfreeze_train_epochs)
        unfreeze_batch_size = kwargs.get('unfreeze_batch_size', self.config.unfreeze_batch_size)
        unfreeze_lr = kwargs.get('unfreeze_lr', self.config.unfreeze_lr)
        initial_weight_path = kwargs.get('initial_weight_path', self.config.pre_training_weights_keras_path)
        use_tensorboard = kwargs.get('use_tensorboard', self.config.use_tensorboard)
        use_checkpoint = kwargs.get('use_checkpoint', self.config.use_checkpoint)
        val_split = kwargs.get('val_split', self.config.val_split)
        use_reduce_lr = kwargs.get('use_reduce_lr', self.config.use_reduce_lr)
        reduce_lr_monitor = kwargs.get('reduce_lr_monitor', self.config.reduce_lr_monitor)
        reduce_lr_factor = kwargs.get('reduce_lr_factor', self.config.reduce_lr_factor)
        reduce_lr_patience = kwargs.get('reduce_lr_patience', self.config.reduce_lr_patience)
        use_early_stopping = kwargs.get('use_early_stopping', self.config.use_early_stopping)
        early_stopping_monitor = kwargs.get('early_stopping_monitor', self.config.early_stopping_monitor)
        early_stopping_min_delta = kwargs.get('early_stopping_min_delta', self.config.early_stopping_min_delta)
        early_stopping_patience = kwargs.get('early_stopping_patience', self.config.early_stopping_patience)

        is_tiny_version = len(self.anchors) == 6  # default setting
        num_classes = len(self.class_names)
        if is_tiny_version:
            model = create_tiny_model(self.model_image_size, self.anchors, num_classes,
                                      freeze_body=2, weights_path=initial_weight_path)
        else:
            model = create_model(self.model_image_size, self.anchors, num_classes,
                                 freeze_body=2,
                                 weights_path=initial_weight_path)  # make sure you know what you freeze

        logger.info('Prepare to train the model...')

        callbacks = []
        if use_tensorboard:
            logging = TensorBoard(log_dir=tensorboard_log_path)
            callbacks.append(logging)
        if use_checkpoint:
            checkpoint = ModelCheckpoint(
                tensorboard_log_path + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                monitor='val_loss', save_weights_only=True, save_best_only=True)
            callbacks.append(checkpoint)

        logger.info('Split dataset for validate...')
        with open(dataset_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val

        logger.info('The first step training begins({} epochs).'.format(frozen_train_epochs))
        # Train with frozen layers first, to get a stable loss.
        # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
        if frozen_train:
            model.compile(optimizer=Adam(lr=frozen_lr), loss={
                # use custom yolo_loss Lambda layer.
                'yolo_loss': lambda y_true, y_pred: y_pred})

            batch_size = frozen_batch_size
            logger.info(
                'Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit(
                self.data_generator_wrapper(lines[:num_train], batch_size, self.model_image_size, self.anchors,
                                            num_classes),
                steps_per_epoch=max(1, num_train // batch_size),
                validation_data=self.data_generator_wrapper(lines[num_train:], batch_size, self.model_image_size,
                                                            self.anchors,
                                                            num_classes),
                validation_steps=max(1, num_val // batch_size),
                epochs=frozen_train_epochs,
                initial_epoch=0,
                callbacks=callbacks)

        logger.info('The second step training begins({} epochs).'.format(unfreeze_train_epochs))
        # Unfreeze and continue training, to fine-tune.
        # Train longer if the result is not good.
        if use_reduce_lr:
            reduce_lr = ReduceLROnPlateau(monitor=reduce_lr_monitor, factor=reduce_lr_factor,
                                          patience=reduce_lr_patience, verbose=1)
            callbacks.append(reduce_lr)
        if use_early_stopping:
            early_stopping = EarlyStopping(monitor=early_stopping_monitor, min_delta=early_stopping_min_delta,
                                           patience=early_stopping_patience, verbose=1)
            callbacks.append(early_stopping)
        if unfreeze_train:
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=Adam(lr=unfreeze_lr),
                          loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
            logger.info('Unfreeze all of the layers.')

            batch_size = unfreeze_batch_size  # note that more GPU memory is required after unfreezing the body
            logger.info('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val,
                                                                                             batch_size))
            model.fit(
                self.data_generator_wrapper(lines[:num_train], batch_size, self.model_image_size, self.anchors,
                                            num_classes),
                steps_per_epoch=max(1, num_train // batch_size),
                validation_data=self.data_generator_wrapper(lines[num_train:], batch_size, self.model_image_size,
                                                            self.anchors,
                                                            num_classes),
                validation_steps=max(1, num_val // batch_size),
                epochs=frozen_train_epochs + unfreeze_train_epochs,
                initial_epoch=unfreeze_train_epochs,
                callbacks=callbacks)
        model.save_weights(output_model_path)
        logger.info('Training completed!')

    def detect_image(self, img: typing.Union[Image.Image, str]) -> typing.List[
            typing.Tuple[str, int, float, int, int, int, int]]:
        """
        在给定图片上做目标检测并返回检测结果

        Args:
            img: 要检测的图片对象（PIL.Image.Image）或路径(str)

        Returns:
            [[类别名称,类别编号,概率,左上角x值,左上角y值,右下角x值,右下角y值],...]
        """
        # 输入参数兼容str类型的图片路径和Image类型的图片文件
        if isinstance(img, str):
            image = Image.open(img)
        else:
            image = img
        assert isinstance(image, Image.Image)
        start = timer()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.compute_output(image_data, [image.size[1], image.size[0]])

        logger.debug('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        results = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            results.append((predicted_class, int(c), float(score), left, top, right, bottom))
            logger.debug('Class {},Position {}, {}'.format(label, (left, top), (right, bottom)))

        end = timer()
        logger.debug('Cost time {}s'.format(end - start))
        return results

    def draw_image(self, img: typing.Union[Image.Image, str], predicted_results: typing.List[
            typing.Tuple[str, int, float, int, int, int, int]], draw_label=True) -> Image.Image:
        """
        给定一张图片和目标检测结果，将目标检测结果绘制在图片上，并返回绘制后的图片

        Args:
            img: 要检测的图片对象（PIL.Image.Image）或路径(str)
            predicted_results: 目标检测的结果，[[类别名称,类别编号,概率,左上角x值,左上角y值,右下角x值,右下角y值],...]
            draw_label: 是否需要为框框标注类别和概率

        Returns:
            添加了检测结果的图片对象
        """
        import cv2
        # 输入参数兼容str类型的图片路径和Image类型的图片文件
        if isinstance(img, str):
            image = Image.open(img)
        else:
            image = img
        assert isinstance(image, Image.Image)

        img_array = np.asarray(image.convert('RGB'))
        for predicted_class, c, score, x1, y1, x2, y2 in predicted_results:
            color = self.colors[c]
            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
            if draw_label:
                label = '{} {:.2f}'.format(predicted_class, score)
                cv2.putText(img_array, text=label, org=(x2 + 3, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.50, color=color, thickness=2)
        image = Image.fromarray(img_array)
        return image

    def detect_and_draw_image(self, image: typing.Union[Image.Image, str], draw_label=True) -> Image.Image:
        """
        在给定图片上做目标检测，并根据检测结果在图片上画出框框和标签

        Args:
            image: 要检测的图片对象（PIL.Image.Image）或路径(str)
            draw_label: 是否需要为框框标注类别和概率

        Returns:
            添加了检测结果的图片对象
        """
        predicted_results = self.detect_image(image)
        img = self.draw_image(image, predicted_results, draw_label=draw_label)
        return img

    def detect_video(self, video_path, output_path=""):
        import cv2
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != "" else False
        if isOutput:
            print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            return_value, frame = vid.read()
            image = Image.fromarray(frame)
            image = self.detect_and_draw_image(image)
            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
