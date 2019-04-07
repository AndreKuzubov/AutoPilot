import os
import sys

from keras.preprocessing import image
import numpy as np
import keras
import RoadSign.utils.utils as utils
from keras.models import load_model
import tensorflow as tf

RESNET50_PATH = "log/models/RESNET50_{input_size}.h5"

def getModel(inputSize, classesCount=1000):
    """

    Получение модели ИНС для обучения.

    :param inputSize: size for inputing images. min valid value = 32
            ожидается в виде массива [sizeW,sizeH,channelsCount]
    :return:
    """
    path = RESNET50_PATH.format(input_size=str(inputSize))
    if (os.path.exists(path)):
        return load_model(path)

    model = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet',
                                                 input_tensor=keras.Input(shape=inputSize),
                                                 input_shape=None, pooling=None, classes=classesCount)
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
    return keras.applications.resnet50.decode_predictions(predictedClasses, top=top)


def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.resnet50.preprocess_input(x)
    preds = model.predict(x)

    return preds