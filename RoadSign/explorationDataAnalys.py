from matplotlib import gridspec as gridspec
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw
import numpy as np
import glob
import tensorflow as tf
import os
from RoadSign import imageProccessing
from RoadSign import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO проверить соответсвие в аналитике с действительностью
def datasetStatistic(picdataset):
    for item in picdataset:
        item["shortList"] = [random.choice(item['images']) for i in range(0, 5)]
        item['glueList'] = imageProccessing.glueImagesHorisontal(
            images=[Image.open(imageFile) for imageFile in item['shortList']],
            size=(30, 30)
        )

    gs = gridspec.GridSpec(len(picdataset), 2, wspace=0.01, hspace=0.1)
    fig = plt.figure(figsize=(7, 20))
    ax1, axBars = [plt.subplot(gs[:, i]) for i in range(2)]
    ax1.axis('off')

    rect = axBars.barh(
        [d['name'] for d in picdataset],
        [len(d['images']) for d in picdataset], color="blue", height=0.8)
    axBars.set_title('DataSet Analytics')
    axBars.set_xlabel("count")
    axBars.set_ylim([-0.45, len(picdataset) - 0.55])
    # axBars.grid(True)

    def autolabelH(rects, ax):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            offset = max([rect.get_width() for rect in rects]) * 0.1
            ax.text(rect.get_width() + offset, rect.get_y() + rect.get_height() / 2,
                    str(rect.get_width()),
                    ha='center', va='center', color="black")

    autolabelH(rect, axBars)

    axPics = [plt.subplot(gs[i, 0]) for i in range(len(picdataset))]
    for i, axPic in enumerate(axPics):
        axPic.axis('off')
        axPic.imshow(picdataset[i]['glueList'])

    utils.createNoExistsFolders("RoadSign/dataanalalysis/GTSRB")
    fig.savefig("RoadSign/dataanalalysis/GTSRB/datasetStatistic1.png")
    fig.show()

    pass


if __name__ == '__main__':
    dataset = {}

    dataSetImages = []
    folder = "RoadSign/datasets/GTSRB/Training/"
    for path in list(next(os.walk(folder)))[1]:
        dataSetImages += [
            {
                "images": [
                    fileName for fileName in glob.glob(folder + path + "/*.ppm")
                ],
                "name": path,
            }
        ]
    datasetStatistic(dataSetImages)
