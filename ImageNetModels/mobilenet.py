import os
import sys

from keras.preprocessing import image
import numpy as np
import keras
import RoadSign.utils.utils as utils
from keras.models import load_model
import tensorflow as tf

MobileNet_PATH = "log/models/MobileNet_{input_size}_{classes}.h5"


def getModel(inputSize, classesCount=1000):
    """

    Получение модели ИНС для обучения.

    :param inputSize: size for inputing images. min valid value = 75
            ожидается в виде массива [sizeW,sizeH,channelsCount]
    :return:
    """
    path = MobileNet_PATH.format(input_size=str(inputSize), classes=str(classesCount))
    if (os.path.exists(path)):
        return load_model(path)

    model = keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3,
                                                   input_tensor=keras.Input(shape=inputSize),
                                                   include_top=True,
                                                   weights='imagenet' if classesCount == 1000 else None,
                                                   pooling=None, classes=classesCount)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    model.save(path)
    return model


def decodeClasses(predictedClasses, top=3, customClasses=None):
    """
    декодирование полученных классов по предсказанным этой моделью.

    :param predictedClasses:
    :param customClasses: должен всегда использоватся, если модель обучена не на imagenet
    :return:
    """
    if (not customClasses is None):
        decodedClasses = []
        for predictClass in predictedClasses:
            mapClasses = [{
                "className": ("i" + str(i), customClasses[i], pred),
                "pred": pred
            } for i, pred in enumerate(predictClass)]
            mapClassesBest = sorted(mapClasses, key=lambda k: k['pred'], reverse=True)
            decodedClasses += [[cl["className"] for cl in mapClassesBest[:top]]]
        return decodedClasses

    return keras.applications.mobilenet.decode_predictions(predictedClasses, top=top)

def preprocess_images(imgs):
    x = np.array([image.img_to_array(img) for img in imgs])
    return keras.applications.xception.preprocess_input(x)

def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.mobilenet.preprocess_input(x)
    preds = model.predict(x)

    return preds