import os
import shutil
import sys

from RoadSign.utils.specificFixs import *
from RoadSign.utils.inetUtils import *
import requests
import lxml.html as lxmlHtml
from urllib.parse import urljoin
from RoadSign.utils import utils
import zipfile
import glob
from PIL import Image
import glob
from RoadSign.utils import imageProccessing
from RoadSign.utils import utils
import random
import numpy as np
import csv
import uuid

#
# Загрузка шаблонов с сайта
#
#
#  @Andreykuzubov 2018

IMAGE_SIZE = 100
BASE_FOLDER = os.getcwd()

GTSRB_DIR = "datasets/GTSRB/"
DORZNAKI_DIR = "datasets/DorZnaki/"
BACKGROUND_DIR = "datasets/background/"
DORZNAKI_PREPARED_DIR = "datasets/DorZnaki_Prepared/"
BACKGROUND_IMAGES = glob.glob(BACKGROUND_DIR + "*.jpg") + glob.glob(BACKGROUND_DIR + "*.png")

BACKGROUND_IMAGES_BUFFER = 300

backgroundImages = []


def loadGTSRB():
    if (not os.path.exists(GTSRB_DIR + "GTSRB_Final_Training_Images")):
        print("loading GTSRB_Final_Training_Images...")
        download_file("http://benchmark.ini.rub.de/Datasets/GTSRB_Final_Training_Images.zip",
                      local_filename=GTSRB_DIR + "GTSRB_Final_Training_Images.zip")

        print("unzipping GTSRB_Final_Training_Images...")
        zip_ref = zipfile.ZipFile(GTSRB_DIR + "GTSRB_Final_Training_Images.zip", 'r')
        zip_ref.extractall(GTSRB_DIR + "GTSRB_Final_Training_Images/")
        zip_ref.close()
        os.remove(GTSRB_DIR + "GTSRB_Final_Training_Images.zip")

        print("success GTSRB_Final_Training_Images")

    if (not os.path.exists(GTSRB_DIR + "GTSRB_Final_Test_Images")):
        print("loading GTSRB_Final_Test_Images...")
        download_file("http://benchmark.ini.rub.de/Datasets/GTSRB_Final_Test_Images.zip",
                      local_filename=GTSRB_DIR + "GTSRB_Final_Test_Images.zip")

        print("unzipping GTSRB_Final_Test_Images...")
        zip_ref = zipfile.ZipFile(GTSRB_DIR + "GTSRB_Final_Test_Images.zip", 'r')
        zip_ref.extractall(GTSRB_DIR + "GTSRB_Final_Test_Images/")
        zip_ref.close()
        os.remove(GTSRB_DIR + "GTSRB_Final_Test_Images.zip")

        print("success GTSRB_Final_Test_Images")


def loadDorZnaki():
    baseUrl = "http://www.artpatch.ru/dorznaki.html"
    html = requests.get(baseUrl).text.replace("<br>", '\n <br>')
    s = lxmlHtml.fromstring(html)

    def loadToLocal(category, name, url):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        utils.createNoExistsFolders(DORZNAKI_DIR + "{category}".format(category=category, name=name))
        with open(DORZNAKI_DIR + "{category}/{name}.zip".format(category=category, name=name),
                  'wb') as handle:
            for block in response.iter_content(1024):
                handle.write(block)

        zip_ref = zipfile.ZipFile(DORZNAKI_DIR + "{category}/{name}.zip".format(category=category, name=name), 'r')

        utils.createNoExistsFolders(DORZNAKI_DIR + "{category}/zip".format(category=category, name=name))
        zip_ref.extractall(DORZNAKI_DIR + "{category}/zip".format(category=category, name=name))
        zip_ref.close()

        filename = glob.glob(DORZNAKI_DIR + "{category}/zip/*".format(category=category))[0]
        os.rename(filename, DORZNAKI_DIR + "{category}/{name}.ai".format(category=category, name=name))

        shutil.rmtree(DORZNAKI_DIR + "{category}/zip".format(category=category))
        os.remove(DORZNAKI_DIR + "{category}/{name}.zip".format(category=category, name=name))
        print("loaded {name}".format(name=name))

    content = s.xpath('//*[@id="content"]')[0]
    category = ""
    url = ""
    for child in content.getchildren()[7:]:
        if (child.tag.lower() == "h2"):
            category = child.text
            if category is None:
                continue
            category = category.replace(" ", "_")
        elif (child.tag.lower() == "br"):
            continue
        elif (list(child.classes)[0] == 'znSootLeft'):
            url = urljoin(base=baseUrl, url=child.getchildren()[0].attrib['href'])
        elif (list(child.classes)[0] == 'znSootRight1'):
            name = child.text_content().replace('\n', '')
            name = name.replace(' ', '_')
            if not os.path.exists(
                    DORZNAKI_DIR + "{category}/{name}.ai".format(category=category, name=name)):
                loadToLocal(category=category, url=url, name=name)

            if not os.path.exists(
                    DORZNAKI_DIR + "{category}/{name}.png".format(category=category, name=name)):
                bash_script = """magick convert ai:'{basefolder}/{filesource}'  -resize '{size}x' -density {density} '{basefolder}/{filedestination}'"""
                os.system(
                    bash_script.format(
                        basefolder=BASE_FOLDER,
                        filesource=DORZNAKI_DIR + "{category}/{name}.ai".format(category=category,
                                                                                name=name),
                        filedestination=DORZNAKI_DIR + "{category}/{name}.png".format(category=category,
                                                                                      name=name),
                        size=str(IMAGE_SIZE),
                        density=str(100))
                )
                print("files created: " + "{name}.png".format(category=category, name=name))


def loadBackGroundImages():
    print("loading backgrounds...")
    download_file("https://storage.googleapis.com/openimages/2018_04/image_ids_and_rotation.csv",
                  local_filename=BACKGROUND_DIR + "image_ids_and_rotation.csv")

    with open(BACKGROUND_DIR + "image_ids_and_rotation.csv", "r") as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader)
        for row in datareader:
            url = row[2]
            name = url[url.rfind("/") + 1:]
            print("loading backgrounds... %s" % (name,))
            download_file(url, local_filename=BACKGROUND_DIR + name)


def generateBackgroundImage(imageSize=IMAGE_SIZE, transparent=None):
    """

    :param imageSize:
    :param transparent: None - не делать получпрозрачным
             иначе максимальное значение прозрачности (0-1)
    :return:
    """
    global backgroundImages

    def popFromBuffer():
        img = backgroundImages[0]
        del backgroundImages[0]

        if (not transparent is None):
            pixeldata = list(img.getdata())
            for i, pixel in enumerate(pixeldata):
                pixeldata[i] = pixel[:3] + (int(random.random() * transparent * 255),)

            img.putdata(pixeldata)

        return img

    if len(backgroundImages) > 0:
        return popFromBuffer()

    while len(backgroundImages) < BACKGROUND_IMAGES_BUFFER:
        file = random.choice(glob.glob(BACKGROUND_DIR + "*.jpg") + glob.glob(BACKGROUND_DIR + "*.png"))
        try:
            im = Image.open(file).convert("RGBA")

            backgroundImages += [im]
        except:
            print("remove %s" % (file))
            os.remove(file)

    for i in range(len(backgroundImages)):
        im = backgroundImages[i]
        cropsize = (int(random.random() * im.size[0]), int(random.random() * im.size[1]))
        cropPoint = (
            int(random.random() * (im.size[0] - cropsize[0])), int(random.random() * (im.size[1] - cropsize[1])))
        im = im.crop((cropPoint[0], cropPoint[1], cropPoint[0] + cropsize[0], cropPoint[1] + cropsize[1]))
        im = im.resize((imageSize, imageSize))
        backgroundImages[i] = im

    return popFromBuffer()


def generateDorZnakiBanches(class_image_count=50000):
    MIN_SIZE_SCALE = 0.7

    classes_files = glob.glob(DORZNAKI_DIR + "Запрещающие_знаки/*.png") \
                    + glob.glob(DORZNAKI_DIR + "Предупреждающие_знаки/*.png") \
                    + glob.glob(DORZNAKI_DIR + "Предписывающие_знаки/*.png") \
                    + glob.glob(DORZNAKI_DIR + "Знаки_приоритета/*.png") \
                    + glob.glob(DORZNAKI_DIR + "Знаки_особых_предписаний/*.png")

    random.shuffle(classes_files)

    for class_file in classes_files:
        class_name = class_file[class_file.rfind("/") + 1:-4]
        print("generate for class %s" % (class_name))
        if (os.path.exists(DORZNAKI_PREPARED_DIR + class_name)):
            continue

        utils.createNoExistsFolders(DORZNAKI_PREPARED_DIR + class_name)

        imagesBuffer = []
        sourceImage = Image.open(class_file).convert("RGBA")
        for i in range(class_image_count):
            if i % 100 == 0:
                print("i = %s" % (i))

            base_img = generateBackgroundImage()

            image = sourceImage.copy().convert("RGBA")

            ratio = random.uniform(0.7, 2)
            image = image.resize(
                (int(IMAGE_SIZE / ratio), int(IMAGE_SIZE * ratio)),
                Image.ANTIALIAS)
            image = image.rotate(random.random() * 90. - 45.)
            ratio = random.uniform(0.5, 4)
            image = image.resize(
                (int(IMAGE_SIZE / ratio), int(IMAGE_SIZE * ratio)),
                Image.ANTIALIAS)
            # darken, lighten
            image = imageProccessing.boxFilter(image, boxSize=1, boxScalar=random.uniform(0.5, 1.5), padding="SAME")

            sign_size = int(random.uniform(MIN_SIZE_SCALE, 1.) * IMAGE_SIZE)
            image = image.resize((sign_size, sign_size), Image.ANTIALIAS)
            paste_point = (
                int(random.random() * (IMAGE_SIZE - sign_size)),
                int(random.random() * (IMAGE_SIZE - sign_size)))
            base_img.paste(image, box=(paste_point), mask=image)

            blurBoxSize = int(random.random() * 3)
            if blurBoxSize > 0:
                base_img = imageProccessing.boxFilter(base_img, boxSize=blurBoxSize, padding="SAME")

            # color correction, dirty window
            base_img.alpha_composite(generateBackgroundImage(transparent=random.random() * 0.4))

            imagesBuffer += [base_img]

            # sourceImage.show()
            # image.show()
            # base_img.show()
            if (len(imagesBuffer) > BACKGROUND_IMAGES_BUFFER):
                for im in imagesBuffer:
                    im.save(DORZNAKI_PREPARED_DIR + class_name + "/" + str(uuid.uuid4()) + ".png")
                imagesBuffer = []

        for im in imagesBuffer:
            im.save(DORZNAKI_PREPARED_DIR + class_name + "/" + str(uuid.uuid4()) + ".png")
        imagesBuffer = []


if __name__ == "__main__":
    # loadBackGroundImages()
    # loadGTSRB()

    # loadDorZnaki()
    generateDorZnakiBanches(1000)
