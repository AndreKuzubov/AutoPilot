import sys
import os

from keras.preprocessing import image
import glob
from PIL import Image
from RoadSign.utils import imageProccessing
from RoadSign.utils import keras_image_processing
import numpy as np
import matplotlib.pyplot as plt

import ImageNetModels.mobilenet as mobilenet
import ImageNetModels.mobilenetv2 as mobilenetv2
import ImageNetModels.xception as xception
import ImageNetModels.inceptionv3 as inceptionv3
import ImageNetModels.densenet as densenet
import ImageNetModels.nasnet as nasnet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# этапы
# обработка всех исходных изображений в единых размер
# предсказание каждой моделью по картинкам

IMAGE_SIZE = 100

SOURCE_IMAGE_FOLDER = "datasets/sourceimages"
SCALED_IMAGE_FOLDER = "datasets/scaledimages_{size}".format(size=str(IMAGE_SIZE))
PROCESSED_IMAGE_FOLDER = "datasets/processedimages_{model}"

TEST_MODELS = [
    ["xception", xception],
    ["inceptionv3", inceptionv3],
    ["mobilenet", mobilenet],
    ["mobilenetv2", mobilenetv2],
    ["densenet", densenet],
    ["nasnet", nasnet],
]

if not os.path.exists(SCALED_IMAGE_FOLDER):
    os.makedirs(SCALED_IMAGE_FOLDER)

if __name__ == "__main__":
    # print("scaling images...")
    # for sourceimage_file in glob.glob(SOURCE_IMAGE_FOLDER + "/" + "*"):
    #     imageName = sourceimage_file[sourceimage_file.rfind("/") + 1:sourceimage_file.rfind(".")]
    #
    #     img = Image.open(sourceimage_file).convert("RGBA")
    #     outimage = keras_image_processing.scallingImage(img, size=[IMAGE_SIZE, IMAGE_SIZE])
    #     outimage.save(SCALED_IMAGE_FOLDER + "/" + imageName + ".png")

    for modelSetting in TEST_MODELS:
        print("modeltesting: %s..." % (modelSetting[0]))

        processedImagesByModel = PROCESSED_IMAGE_FOLDER.format(model=modelSetting[0])

        if not os.path.exists(processedImagesByModel):
            os.makedirs(processedImagesByModel)

        model = modelSetting[1].getModel(inputSize=[IMAGE_SIZE, IMAGE_SIZE, 3])
        imgs = glob.glob(SCALED_IMAGE_FOLDER + "/" + "*")
        for imageIndex, sourceimage_file in enumerate(imgs):
            imageName = sourceimage_file[sourceimage_file.rfind("/") + 1:sourceimage_file.rfind(".")]
            img = Image.open(sourceimage_file).convert("RGB")

            pred = modelSetting[1].predict(model, img)
            predClasses = modelSetting[1].decodeClasses(pred, top=2)
            print("imageName = %s" % str(['-'.join(str(p) for p in c) for c in predClasses[0]]))

            fig = plt.gca()
            fig.axis('off')
            fig.set_title("\n".join(['-'.join(str(p) for p in c) for c in predClasses[0]]))
            fig.imshow(img)
            plt.savefig(processedImagesByModel + "/" + imageName + ".png")
