from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def wbImage(image):
    x = tf.constant(np.asarray(image), dtype=tf.float32)
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


def glueImagesHorisontal(images, size=(30, 30)):
    widths, heights = size

    total_width = int(widths * len(images))
    total_height = heights

    new_im = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    for im in images:
        im = im.resize(size)
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im
