import os
import shutil

import requests
import lxml.html as lxmlHtml
from urllib.parse import urljoin
from RoadSign.utils import utils
import zipfile
import glob

# Загрузка шаблонов с сайта
#
#  @Andreykuzubov 2018

def getPatterns():
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



if __name__ == "__main__":
    getPatterns()
