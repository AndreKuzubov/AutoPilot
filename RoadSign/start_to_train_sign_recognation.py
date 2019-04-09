import os
import sys

from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
import glob
import random
from PIL import Image
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

SOURCE_IMAGES_GROP_MASK = "datasets/DorZnaki_Prepared/*/*.png"
SOURCE_IMAGES_CLASSES = [f[f.rfind("/") + 1:] for f in glob.glob("datasets/DorZnaki_Prepared/*")]
TRAINED_MODEL_FOLDER = "log/models/{model}_{inputsize}_{classes}.h5"
IMAGE_INPUT_SIZE = [100, 100, 3]

TEST_MODELS = [
    ["xception", xception],
    ["inceptionv3", inceptionv3],
    ["mobilenet", mobilenet],
    ["mobilenetv2", mobilenetv2],
    ["densenet", densenet],
    # ["nasnet", nasnet],
]


def next_batch(batch_size):
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
        labels += [y]

    return images, labels


if __name__ == "__main__":
    for modelSetting in TEST_MODELS:
        print("modeltraining: %s..." % (modelSetting[0]))
        model = modelSetting[1].getModel(inputSize=IMAGE_INPUT_SIZE, classesCount=len(SOURCE_IMAGES_CLASSES))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        saveModelPath = TRAINED_MODEL_FOLDER.format(model=modelSetting[0], inputsize=str(IMAGE_INPUT_SIZE),
                                                    classes=str(len(SOURCE_IMAGES_CLASSES)))

        for i in range(5):
            batch_images, batch_labels = next_batch(batch_size=10000)
            batch_x = modelSetting[1].preprocess_images(batch_images)
            batch_y = np.array(batch_labels)

            # train
            model.fit(batch_x, batch_y,
                      verbose=1,
                      epochs=1,
                      # validation_data=(test_xs, test_ys),
                      validation_split=0.1,
                      callbacks=[ModelCheckpoint(filepath=saveModelPath,
                                                 monitor="val_acc",
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 mode="auto")]
                      )
        model.save(saveModelPath)

    for modelSetting in TEST_MODELS:
        batch_images, batch_labels = next_batch(batch_size=100)
        batch_x = modelSetting[1].preprocess_images(batch_images)
        batch_y = np.array(batch_labels)

        print("modeltesting: %s..." % (modelSetting[0]))
        loadModelPath = TRAINED_MODEL_FOLDER.format(model=modelSetting[0], inputsize=str(IMAGE_INPUT_SIZE),
                                                    classes=str(len(SOURCE_IMAGES_CLASSES)))
        model = load_model(loadModelPath)
        score = model.evaluate(batch_x, batch_y, verbose=0)
        print("Model: %s Test score: %f Test accuracy: %f " % (modelSetting[0], score[0], score[1]))
