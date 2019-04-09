# обучение основных архитектур компьютерного зрения на базе дорожных знаков
#
# AndreyKuzubov
#
import datetime
import os
import sys

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
import ImageNetModels.mobilenet as mobilenet
import ImageNetModels.mobilenetv2 as mobilenetv2
import ImageNetModels.xception as xception
import ImageNetModels.inceptionv3 as inceptionv3
import ImageNetModels.densenet as densenet
import ImageNetModels.nasnet as nasnet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TIME_TAG = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

SOURCE_IMAGES_GROP_MASK = "datasets/DorZnaki_Prepared/*/*.png"
SOURCE_IMAGES_CLASSES = sorted([f[f.rfind("/") + 1:] for f in glob.glob("datasets/DorZnaki_Prepared/*")])

PROCESSED_IMAGES_FOLDER = "log/images/{date}/{model}"
TRAINED_MODEL_FOLDER = "log/models/{model}_{inputsize}_{classes}.h5"
TENSOR_BOARD_FOLDER = "log/tensorboard/{date}/{model}"
IMAGE_INPUT_SIZE = [100, 100, 3]

TEST_MODELS = [
    ["xception", xception],
    ["inceptionv3", inceptionv3],
    ["mobilenet", mobilenet],
    ["mobilenetv2", mobilenetv2],
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
        label = label[label.rfind("/") + 1:]
        images += [Image.open(imgFile).convert("RGB")]

        y = [0] * len(SOURCE_IMAGES_CLASSES)
        y[SOURCE_IMAGES_CLASSES.index(label)] = 1
        labels += [label] if yAsNames else [y]

    return images, labels


if __name__ == "__main__":
    for modelSetting in TEST_MODELS:
        print("modeltraining: %s..." % (modelSetting[0]))
        model = modelSetting[1].getModel(inputSize=IMAGE_INPUT_SIZE, classesCount=len(SOURCE_IMAGES_CLASSES))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        saveModelPath = TRAINED_MODEL_FOLDER.format(model=modelSetting[0], inputsize=str(IMAGE_INPUT_SIZE),
                                                    classes=str(len(SOURCE_IMAGES_CLASSES)))
        logTrainPath = TENSOR_BOARD_FOLDER.format(date=TIME_TAG, model=modelSetting[0])
        if not os.path.exists(logTrainPath):
            os.makedirs(logTrainPath)

        for i in range(5):
            batch_images, batch_labels = next_batch(batch_size=10000)
            batch_x = modelSetting[1].preprocess_images(batch_images)
            batch_y = np.array(batch_labels)

            session = model.fit(batch_x, batch_y,
                                verbose=1,
                                epochs=1,
                                # validation_data=(test_xs, test_ys),
                                validation_split=0.1,
                                callbacks=[
                                    ModelCheckpoint(filepath=saveModelPath,
                                                    monitor="val_acc",
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    mode="auto"),
                                    TensorBoard(log_dir=logTrainPath,
                                                histogram_freq=0,
                                                write_graph=True,
                                                write_grads=True,
                                                write_images=True)
                                ])
        model.save(saveModelPath)

    for modelSetting in TEST_MODELS[:1]:
        print("modeltesting: %s..." % (modelSetting[0]))
        processedImagesByModelFolder = PROCESSED_IMAGES_FOLDER.format(date=TIME_TAG, model=modelSetting[0])
        logTrainPath = TENSOR_BOARD_FOLDER.format(date=TIME_TAG, model=modelSetting[0])
        loadModelPath = TRAINED_MODEL_FOLDER.format(model=modelSetting[0], inputsize=str(IMAGE_INPUT_SIZE),
                                                    classes=str(len(SOURCE_IMAGES_CLASSES)))
        if not os.path.exists(logTrainPath):
            os.makedirs(logTrainPath)
        if not os.path.exists(processedImagesByModelFolder):
            os.makedirs(processedImagesByModelFolder)

        model = load_model(loadModelPath)

        batch_images, batch_labels = next_batch(batch_size=100)
        batch_x = modelSetting[1].preprocess_images(batch_images)
        batch_y = np.array(batch_labels)
        score = model.evaluate(batch_x, batch_y, verbose=0)
        print("Model: %s Test score: %f Test accuracy: %f " % (modelSetting[0], score[0], score[1]))

        imgs, img_names = next_batch(batch_size=20, yAsNames=True)
        batch_imgs_tensor = modelSetting[1].preprocess_images(imgs)

        for imageIndex, img in enumerate(imgs):
            imageName = img_names[imageIndex]
            imageNameEn = translit(imageName, "ru", reversed=True)

            pred = modelSetting[1].predict(model, img)
            predClasses = list(modelSetting[1].decodeClasses(pred, top=2, customClasses=SOURCE_IMAGES_CLASSES))
            print("imageName = %s" % str(['-'.join(str(p) for p in c) for c in predClasses[0]]))
            predictedImageNet = translit(str(['-'.join(str(p) for p in c) for c in predClasses[0]]), "ru",
                                         reversed=True)
            tf.summary.image(imageNameEn + "----->" + predictedImageNet,
                             np.expand_dims(batch_imgs_tensor[imageIndex], axis=0))

            fig = plt.gca()
            fig.axis('off')
            fig.set_title("\n".join(['-'.join(str(p) for p in c) for c in predClasses[0]]))
            fig.imshow(img)
            plt.savefig(processedImagesByModelFolder + "/{index}_{imagename}.png".format(imagename=imageName,
                                                                                         index=str(imageIndex)))
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(logdir=logTrainPath)
        all_summaries = tf.summary.merge_all()
        summary = sess.run(all_summaries)
        summary_writer.add_summary(summary=summary)
        summary_writer.close()
