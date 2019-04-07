"""
source: https://github.com/AndreKuzubov/TensorFlowDecisions

"""

import sys
import os

sys.path.append('../')
import tensorflow as tf
import numpy as np
from PIL import Image

from keras.models import Sequential
from keras.layers import Cropping2D, Lambda

def scroppingImage(sourceImage, vertical_cropping=(2, 2), horizontal_croping=(4, 4)):
    """

    :param sourceImage:
    :param vertical_cropping: кол-во пикселей обрезать сверху и снизу
    :param horizontal_croping: кол-во пикселей обрезать слева и справа
    :return:
    """
    model = Sequential()
    model.add(Cropping2D(cropping=(vertical_cropping, horizontal_croping)))

    xImage = np.asarray(sourceImage)
    inputShape = xImage.shape
    xImage = xImage.reshape([1, inputShape[0], inputShape[1], inputShape[2]])

    yImage = model.predict(xImage)
    outputShape = yImage.shape
    yImage = yImage.reshape([outputShape[1], outputShape[2], outputShape[3]])

    return Image.fromarray(yImage.astype(np.uint8))


def scallingImage(sourceImage, size=(256, 256)):
    model = Sequential()
    model.add(Lambda(lambda image: tf.image.resize_images(image, size)))

    xImage = np.asarray(sourceImage)
    inputShape = xImage.shape
    xImage = xImage.reshape([1, inputShape[0], inputShape[1], inputShape[2]])

    yImage = model.predict(xImage)
    outputShape = yImage.shape
    yImage = yImage.reshape([outputShape[1], outputShape[2], outputShape[3]])
    return Image.fromarray(yImage.astype(np.uint8))


if __name__ == "__main__":
    source = Image.open("log/source_image.jpg")

    copped_image = scroppingImage(sourceImage=source, vertical_cropping=(100, 100), horizontal_croping=(200, 400))
    copped_image.show()
    copped_image.save("log/scroppingImage.jpg")

    scalled_image = scallingImage(sourceImage=source)
    scalled_image.show()
    scalled_image.save("log/scallingImage.jpg")
