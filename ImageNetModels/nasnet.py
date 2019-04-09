import os
import sys

from keras.preprocessing import image
import numpy as np
import keras
from keras.models import load_model
import tensorflow as tf

NASNetMobile_PATH = "log/models/NASNetMobile_{input_size}_{classes}.h5"


def getModel(inputSize, classesCount=1000):
    """

    Получение модели ИНС для обучения.

    :param inputSize: size for inputing images. min valid value = 32
            ожидается в виде массива [sizeW,sizeH,channelsCount]
    :return:
    """
    path = NASNetMobile_PATH.format(input_size=str(inputSize), classes=str(classesCount))
    if (os.path.exists(path)):
        return load_model(path)

    model = keras.applications.nasnet.NASNetMobile(input_shape=None, include_top=True,
                                                   weights='imagenet' if classesCount == 1000 else None,
                                                   input_tensor=keras.Input(shape=inputSize),
                                                   pooling=None, classes=classesCount)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    model.save(path)
    return model


def decodeClasses(predictedClasses, top=3):
    """
    декодирование полученных классов по предсказанным этой моделью.

    :param predictedClasses:
    :return:
    """
    return keras.applications.nasnet.decode_predictions(predictedClasses, top=top)

def preprocess_images(imgs):
    x = np.array([image.img_to_array(img) for img in imgs])
    return keras.applications.xception.preprocess_input(x)

def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.nasnet.preprocess_input(x)
    preds = model.predict(x)

    return preds