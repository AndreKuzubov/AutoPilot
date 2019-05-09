# обучение основных архитектур компьютерного зрения на базе дорожных знаков
#
# AndreyKuzubov
#
import datetime
import os
import sys
import time

from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
import glob
import random
from PIL import Image
from tensorflow.python.keras.callbacks import TensorBoard
import tensorflow as tf
from transliterate import translit, get_available_language_codes

from RoadSign.utils import imageProccessing
from RoadSign.utils import keras_image_processing
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

import ImageNetModels
import ImageNetModels.resnet as resnet
import ImageNetModels.mobilenet as mobilenet
import ImageNetModels.mobilenetv2 as mobilenetv2
import ImageNetModels.xception as xception
import ImageNetModels.inceptionv3 as inceptionv3
import ImageNetModels.densenet as densenet
import ImageNetModels.nasnet as nasnet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TIME_TAG = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

SOURCE_IMAGES_GROP_MASK = "datasets/GTSRB_indexed/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/*/*.ppm".replace("/",os.sep)
MODEL_IMAGES_CLASSES = sorted([f[f.rfind("/") + 1:] for f in glob.glob("datasets/DorZnaki_Prepared/*".replace("/",os.sep))])

PROCESSED_IMAGES_FOLDER = "log/images/{date}/{model}".replace("/",os.sep)
TRAINED_MODEL_FOLDER = "log/models/{model}_{inputsize}_{classes}.h5".replace("/",os.sep)
TENSOR_BOARD_FOLDER = "log/tensorboard/{date}/{model}".replace("/",os.sep)
IMAGE_INPUT_SIZE = [100, 100, 3]

TEST_MODELS = [
    ["xception", xception],
    ["inceptionv3", inceptionv3],
    ["mobilenet", mobilenet],
    ["mobilenetv2", mobilenetv2],
    ["resnet", resnet],
    ["densenet", densenet],
    # ["nasnet", nasnet],
]


def next_batch(batch_size, yAsNames=False):
    """

    :param batch_size: размер батча
    :param yAsNames: если  True, тогда labels будут == строковыми значениями (названиеми знаков)
        если False - то в виде массива с одно единицей (значищей класс)
    :return:
        images - картики знаков
        labels - названия или массив классов картинки (зависит от yAsNames)
    """
    imgFiles = glob.glob(SOURCE_IMAGES_GROP_MASK)
    random.shuffle(imgFiles)

    images = []
    labels = []
    for imgFile in imgFiles[:batch_size]:
        label = os.path.dirname(imgFile)
        label = label[label.rfind(os.sep) + 1:]
        images += [Image.open(imgFile).resize(IMAGE_INPUT_SIZE[:2], Image.ANTIALIAS).convert("RGB")]

        y = [0] * len(MODEL_IMAGES_CLASSES)
        y[MODEL_IMAGES_CLASSES.index(label)] = 1
        labels += [label] if yAsNames else [y]

    return images, labels


if __name__ == "__main__":
    for modelSetting in TEST_MODELS:
        print("modeltesting on GTSRB_indexed: %s..." % (modelSetting[0]))
        testBatchSize = 100
        processedImagesByModelFolder = PROCESSED_IMAGES_FOLDER.format(date=TIME_TAG, model=modelSetting[0])
        logTrainPath = TENSOR_BOARD_FOLDER.format(date=TIME_TAG, model=modelSetting[0])
        loadModelPath = TRAINED_MODEL_FOLDER.format(model=modelSetting[0], inputsize=str(IMAGE_INPUT_SIZE),
                                                    classes=str(len(MODEL_IMAGES_CLASSES)))
        if not os.path.exists(logTrainPath):
            os.makedirs(logTrainPath)
        if not os.path.exists(processedImagesByModelFolder):
            os.makedirs(processedImagesByModelFolder)

        model = load_model(loadModelPath)
        print(model.summary())

        batch_images, batch_labels = next_batch(batch_size=testBatchSize)
        batch_x = modelSetting[1].preprocess_images(batch_images)
        batch_y = np.array(batch_labels)
        startTime = time.time()
        score = model.evaluate(batch_x, batch_y, verbose=0)
        endTime = time.time()
        print("Model: %s Test score: %.4f Test accuracy: %.4f SprendTime %.4f for batch %d" % (
            modelSetting[0], score[0], score[1], endTime - startTime, testBatchSize))

        imgs, img_names = next_batch(batch_size=20, yAsNames=True)
        batch_imgs_tensor = modelSetting[1].preprocess_images(imgs)

        for imageIndex, img in enumerate(imgs):
            imageName = img_names[imageIndex]
            imageNameEn = translit(imageName, "ru", reversed=True)

            pred = modelSetting[1].predict(model, img)
            predClasses = list(modelSetting[1].decodeClasses(pred, top=2, customClasses=MODEL_IMAGES_CLASSES))
            print("imageName = %s" % str(['-'.join(str(p) for p in c) for c in predClasses[0]]))
            predictedImageNet = translit(str(['-'.join(str(p) for p in c) for c in predClasses[0]]), "ru",
                                         reversed=True)
            tf.summary.image(imageNameEn + "----->" + predictedImageNet,
                             np.expand_dims(batch_imgs_tensor[imageIndex], axis=0))

            fig = plt.gca()
            fig.axis('off')
            fig.set_title("\n".join(['-'.join(str(p) for p in c) for c in predClasses[0]]))
            fig.imshow(img)
            plt.savefig(processedImagesByModelFolder + "/{index}_{imagename}.png".replace("/",os.sep).format(imagename=imageName,
                                                                                         index=str(imageIndex)))
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(logdir=logTrainPath)
        all_summaries = tf.summary.merge_all()
        summary = sess.run(all_summaries)
        summary_writer.add_summary(summary=summary)
        summary_writer.close()
