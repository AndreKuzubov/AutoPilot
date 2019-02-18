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

#
# Загрузка шаблонов с сайта
#
#
#  @Andreykuzubov 2018

IMAGE_SIZE = 512
BASE_FOLDER = os.getcwd()

GTSRB_DIR = "datasets/GTSRB/"
DORZNAKI_DIR = "datasets/DorZnaki/"

def loadGTSRB():
    if (not os.path.exists(GTSRB_DIR+"GTSRB_Final_Training_Images")):
        print("loading GTSRB_Final_Training_Images...")
        download_file("http://benchmark.ini.rub.de/Datasets/GTSRB_Final_Training_Images.zip",
                      local_filename=GTSRB_DIR+"GTSRB_Final_Training_Images.zip")

        print("unzipping GTSRB_Final_Training_Images...")
        zip_ref = zipfile.ZipFile(GTSRB_DIR+"GTSRB_Final_Training_Images.zip", 'r')
        zip_ref.extractall(GTSRB_DIR+"GTSRB_Final_Training_Images/")
        zip_ref.close()
        os.remove(GTSRB_DIR+"GTSRB_Final_Training_Images.zip")

        print("success GTSRB_Final_Training_Images")

    if (not os.path.exists(GTSRB_DIR+"GTSRB_Final_Test_Images")):
        print("loading GTSRB_Final_Test_Images...")
        download_file("http://benchmark.ini.rub.de/Datasets/GTSRB_Final_Test_Images.zip",
                      local_filename=GTSRB_DIR+"GTSRB_Final_Test_Images.zip")

        print("unzipping GTSRB_Final_Test_Images...")
        zip_ref = zipfile.ZipFile(GTSRB_DIR+"GTSRB_Final_Test_Images.zip", 'r')
        zip_ref.extractall(GTSRB_DIR+"GTSRB_Final_Test_Images/")
        zip_ref.close()
        os.remove(GTSRB_DIR+"GTSRB_Final_Test_Images.zip")

        print("success GTSRB_Final_Test_Images")


def loadDorZnaki():
    baseUrl = "http://www.artpatch.ru/dorznaki.html"
    html = requests.get(baseUrl).text.replace("<br>", '\n <br>')
    s = lxmlHtml.fromstring(html)

    def loadToLocal(category, name, url):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        utils.createNoExistsFolders(DORZNAKI_DIR+"{category}".format(category=category, name=name))
        with open(DORZNAKI_DIR+"{category}/{name}.zip".format(category=category, name=name),
                  'wb') as handle:
            for block in response.iter_content(1024):
                handle.write(block)

        zip_ref = zipfile.ZipFile(DORZNAKI_DIR+"{category}/{name}.zip".format(category=category, name=name), 'r')

        utils.createNoExistsFolders(DORZNAKI_DIR+"{category}/zip".format(category=category, name=name))
        zip_ref.extractall(DORZNAKI_DIR+"{category}/zip".format(category=category, name=name))
        zip_ref.close()

        filename = glob.glob(DORZNAKI_DIR+"{category}/zip/*".format(category=category))[0]
        os.rename(filename, DORZNAKI_DIR+"{category}/{name}.ai".format(category=category, name=name))

        shutil.rmtree(DORZNAKI_DIR+"{category}/zip".format(category=category))
        os.remove(DORZNAKI_DIR+"{category}/{name}.zip".format(category=category, name=name))
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
                    DORZNAKI_DIR+"{category}/{name}.ai".format(category=category, name=name)):
                loadToLocal(category=category, url=url, name=name)

            if not os.path.exists(
                    DORZNAKI_DIR+"{category}/{name}.png".format(category=category, name=name)):
                bash_script = """magick convert ai:'{basefolder}/{filesource}'  -resize '{size}x' -density {density} '{basefolder}/{filedestination}'"""
                os.system(
                    bash_script.format(
                        basefolder=BASE_FOLDER,
                        filesource=DORZNAKI_DIR+"{category}/{name}.ai".format(category=category,
                                                                                   name=name),
                        filedestination=DORZNAKI_DIR+"{category}/{name}.png".format(category=category,
                                                                                         name=name),
                        size=str(IMAGE_SIZE),
                        density=str(100))
                )
                print("files created: " + "{name}.png".format(category=category, name=name))


if __name__ == "__main__":
    loadGTSRB()

    loadDorZnaki()

