from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def wbImage(image):
    x = tf.constant(np.asarray(image), dtype=tf.float32)

    if (x.shape[2] == 4):
        y = tf.constant(np.asarray(
            [
                [1 / 3, 1 / 3, 1 / 3, 0],
                [1 / 3, 1 / 3, 1 / 3, 0],
                [1 / 3, 1 / 3, 1 / 3, 0],
                [0, 0, 0, 1],
            ]
        ), dtype=tf.float32)
    else:
        y = tf.constant(np.asarray(
            [
                [1 / 3, 1 / 3, 1 / 3],
                [1 / 3, 1 / 3, 1 / 3],
                [1 / 3, 1 / 3, 1 / 3],
            ]
        ), dtype=tf.float32)

    # print('x shape:', x.shape)
    # print('y shape:', y.shape)
    out = tf.tensordot(x, y, axes=[[2], [0]])
    with tf.Session() as sess:
        result = sess.run(out)
        sess.close()
        # print('result shape:', result.shape)
        return Image.fromarray(np.array(result, dtype=np.uint8))


def lightingImage(image, ratio):
    x = tf.constant(np.asarray(image), dtype=tf.float32)
    y = tf.constant(np.asarray(
        [ratio, ratio, ratio, 1],
    ), dtype=tf.float32)

    print('x shape:', x.shape)
    print('y shape:', y.shape)
    out = tf.einsum("ijk,k->ijk", x, y)
    with tf.Session() as sess:
        result = sess.run(out)
        sess.close()
        # print('result shape:', result.shape)
        return Image.fromarray(np.array(result, dtype=np.uint8))


def contrastImage(image, ratio):
    x = tf.constant(np.asarray(image), dtype=tf.float32)
    y = tf.constant(1 / 8 * np.asarray(
        [
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, 1]
        ]
    ), dtype=tf.float32)

    print('x shape:', x.shape)
    print('y shape:', y.shape)

    out = tf.nn.conv2d(x, y, padding="VALID", strides=[1, 1, 1, 1])
    with tf.Session() as sess:
        result = sess.run(out)
        sess.close()
        # print('result shape:', result.shape)
        return Image.fromarray(np.array(result, dtype=np.uint8))


def blur(sourceFile, destinationFile, radius=100, sigma=1):
    os.system("magick convert -blur {radius}x{sigma} {source} {destination}"
        .format(
        source=sourceFile,
        destination=destinationFile,
        radius=str(radius),
        sigma=str(sigma)
    ))


def boxFilter(sourceImage, boxSize=3, boxScalar=None, padding="VALID"):
    """
    box фильтр - размытие изображения
    :param sourceImage: исходное изображение
    :param boxScalar: множетель ядра фильтра - определяет яркость полученного зображения
            если boxScalar = 1/boxSize**2 - яркость не меняется
    :param padding: исходное изображение
    :return: обработанное изображение
    """
    if (boxScalar is None):
        boxScalar = 1. / boxSize ** 2

    x_image = tf.constant(np.asarray(sourceImage).astype(np.float32), dtype=tf.float32)
    x_image = tf.transpose(x_image, [2, 0, 1])
    x_image = tf.reshape(x_image, [n.value for n in x_image.shape] + [1])

    kernel = tf.constant(boxScalar * np.asarray([[1] * boxSize] * boxSize), dtype=tf.float32)
    kernel = tf.reshape(kernel, [boxSize, boxSize, 1, 1])

    filtered = tf.nn.conv2d(x_image, kernel, strides=[1, 1, 1, 1], padding=padding)

    with tf.Session() as sess:
        y_image, = sess.run([filtered])

    y_image = y_image.transpose((3, 1, 2, 0,))
    y_image = y_image.reshape(y_image.shape[1:])

    # обработка засветов
    y_image = np.minimum(y_image, 255)
    y_image = np.maximum(y_image, 0)
    newImage = Image.fromarray(y_image.astype(np.uint8), sourceImage.mode)
    return newImage


def glueImagesHorisontal(images, size=(30, 30)):
    widths, heights = size

    total_width = int(widths * len(images))
    total_height = heights

    new_im = Image.new('RGBA', (total_width, total_height))

    x_offset = 0
    for im in images:
        im = im.resize(size)
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im
