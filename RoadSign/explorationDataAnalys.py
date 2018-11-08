from matplotlib import gridspec as gridspec
import matplotlib.pyplot as plt
import random
from PIL import Image
import glob
import os
from RoadSign.utils import imageProccessing
from RoadSign.utils import utils
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEBUG = False

#
# Аналитика обучающих данных
#
#   @AndreyKuzubov 2018

def dataAnalitics(datasetFolder, tag):
    dataSetImages = []
    for path in list(next(os.walk(datasetFolder)))[1]:
        dataSetImages += [
            {
                "images": [
                              fileName for fileName in glob.glob(datasetFolder + path + "/*.ppm")
                          ] + [
                              fileName for fileName in glob.glob(datasetFolder  + path + "/*.png")
                          ],
                "name": path,
            }
        ]
    __datasetStatistic(dataSetImages, tag)


def __datasetStatistic(picdataset, tag):
    def autolabelH(rects, ax):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            offset = max([rect.get_width() for rect in rects]) * 0.1
            ax.text(rect.get_width() + offset, rect.get_y() + rect.get_height() / 2,
                    str(rect.get_width()),
                    ha='center', va='center', color="black")

    for item in picdataset:
        item["shortList"] = [random.choice(item['images']) for i in range(0, 5)]
        item['glueList'] = imageProccessing.glueImagesHorisontal(
            images=[Image.open(imageFile) for imageFile in item['shortList']],
            size=(50, 50)
        )

    __logDataSetStatisticValues(picdataset=picdataset, tag=tag)
    gs = gridspec.GridSpec(len(picdataset), 4, wspace=0.1, hspace=0.1)
    fig = plt.figure(figsize=(7, 20))

    # bars1 - кол-во в наборе
    axBars = plt.subplot(gs[:, 2:4])
    axBars.axis('off')
    rect = axBars.barh(
        [d['name'] for d in picdataset],
        [len(d['images']) for d in picdataset], color="blue", height=0.8)
    axBars.set_title('Count of pictures in DataSet')
    axBars.set_xlabel("count")
    axBars.set_ylim([-0.45, len(picdataset) - 0.55])
    axBars.set_yticks([])
    # axBars.grid(True)
    autolabelH(rect, axBars)

    # визуализация картинок
    axPics = [plt.subplot(gs[i, 1]) for i in range(len(picdataset))]
    for i, axPic in enumerate(axPics):
        axPic.axis('off')
        axPic.imshow(picdataset[i]['glueList'])

    # подпись набора
    axTexts = [plt.subplot(gs[i, 0]) for i in range(len(picdataset))]
    for i, axText in enumerate(axTexts):
        axText.axis('off')
        axText.text(0.5, 0.5, picdataset[i]['name'],
                    horizontalalignment='center',
                    verticalalignment='center')

    utils.createNoExistsFolders("RoadSign/dataanalalysis/{tag}".format(tag=tag))
    fig.savefig("RoadSign/dataanalalysis/{tag}/datasetStatistic1.png".format(tag=tag))
    fig.show()


def __logDataSetStatisticValues(picdataset, tag):
    utils.createNoExistsFolders("RoadSign/dataanalalysis/{tag}".format(tag=tag))
    with open("RoadSign/dataanalalysis/{tag}/datasetStatistics.txt".format(tag=tag), "w+") as file:
        for item in picdataset:
            count = len(item['images'])
            images = [Image.open(imageFile) for imageFile in item['images']]
            stdSize = np.std([max(image.size) for image in images])
            maxSize = max([max(image.size) for image in images])
            minSize = min([min(image.size) for image in images])
            name = item['name']

            if (DEBUG):
                print("{tag} {name} count: {count} stdSize: {stdSize} maxSize: {maxSize} minSize {minSize}".format(
                    tag=tag,
                    name=name,
                    count=str(count),
                    stdSize=str(stdSize),
                    maxSize=str(maxSize),
                    minSize=str(minSize),
                ))
            file.write(
                "{tag} {name} count: {count} stdSize: {stdSize} maxSize: {maxSize} minSize {minSize}".format(
                    tag=tag,
                    name=name,
                    count=str(count),
                    stdSize=str(stdSize),
                    maxSize=str(maxSize),
                    minSize=str(minSize),
                ) + '\n')
        file.close()


if __name__ == '__main__':
    DEBUG = True
    # dataAnalitics(
    #     datasetFolder="RoadSign/datasets/GTSRB/Training/",
    #     tag="GTSRB_original"
    # )
    dataAnalitics(
        datasetFolder="RoadSign/datasets/fromPattern_original/Запрещающие знаки/",
        tag="fromPattern_original/Запрещающие знаки"
    )
