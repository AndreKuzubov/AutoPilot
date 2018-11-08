import os
import shutil

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

def loadPatternsFromSite():
    baseUrl = "http://www.artpatch.ru/dorznaki.html"
    html = requests.get(baseUrl).text.replace("<br>", '\n <br>')
    s = lxmlHtml.fromstring(html)

    def loadToLocal(category, name, url):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        utils.createNoExistsFolders("RoadSign/datasets/patterns/{category}".format(category=category, name=name))
        with open("RoadSign/datasets/patterns/{category}/{name}.zip".format(category=category, name=name),
                  'wb') as handle:
            for block in response.iter_content(1024):
                handle.write(block)

        zip_ref = zipfile.ZipFile(
            "RoadSign/datasets/patterns/{category}/{name}.zip".format(category=category, name=name), 'r')

        utils.createNoExistsFolders("RoadSign/datasets/patterns/{category}/zip".format(category=category, name=name))
        zip_ref.extractall("RoadSign/datasets/patterns/{category}/zip".format(category=category, name=name))
        zip_ref.close()

        filename = glob.glob("RoadSign/datasets/patterns/{category}/zip/*".format(category=category))[0]
        os.rename(filename, "RoadSign/datasets/patterns/{category}/{name}.ai".format(category=category, name=name))

        shutil.rmtree("RoadSign/datasets/patterns/{category}/zip".format(category=category))
        os.remove("RoadSign/datasets/patterns/{category}/{name}.zip".format(category=category, name=name))
        print("loaded {name}".format(name=name))

    content = s.xpath('//*[@id="content"]')[0]
    category = ""
    url = ""
    for child in content.getchildren()[7:]:
        if (child.tag.lower() == "h2"):
            category = child.text
        elif (child.tag.lower() == "br"):
            continue
        elif (list(child.classes)[0] == 'znSootLeft'):
            url = urljoin(base=baseUrl, url=child.getchildren()[0].attrib['href'])
        elif (list(child.classes)[0] == 'znSootRight1'):
            name = child.text_content().replace('\n', '')
            loadToLocal(category=category, url=url, name=name)


def createDataSetFromPatterns():
    bash_script = """magick convert ai:'{basefolder}/{filesource}'  -resize '{size}x' -density {density} '{basefolder}/{filedestination}'"""

    baseFolder = os.getcwd()
    paternsFolder = "RoadSign/datasets/patterns"
    densityFolder = "RoadSign/datasets/fromPattern_original"
    for category in next(os.walk(paternsFolder))[1]:
        for fileName in next(os.walk(paternsFolder + "/" + category))[2]:
            fileSource = paternsFolder + '/' + category + '/' + fileName

            utils.createNoExistsFolders(densityFolder + '/' + category + '/' + fileName[:-3])
            for density in range(100, 500, 50):
                for size in range(50, 500, 50):
                    fileDesitination = densityFolder + '/' + category + '/' + fileName[:-3] + '/' \
                                       + str(density) + "_" + str(size) + '.png'
                    os.system(
                        bash_script.format(
                            basefolder=baseFolder,
                            filesource=fileSource,
                            filedestination=fileDesitination,
                            size=str(size),
                            density=str(density))
                    )
            print("files created: " + fileName)


if __name__ == "__main__":
    # loadPatternsFromSite()
    createDataSetFromPatterns()
