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
