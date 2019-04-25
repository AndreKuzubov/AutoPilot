## Распознавание дорожных знаков 

Распозноваиние дорожных знаков реализовано на основе НС. 

#### Этапы:
1) [Поготовка тренировочных данных](start_createDataSetOnPatters.py): 
  
    * [x] Выгрузка всех классов знаков и их названии
     
        * [x] Набор [<b>GTSRB</b>](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
        * [x] Набор [<b>DorZnaki</b>](http://www.artpatch.ru/dorznaki.html) 
        * [x] Набор фоновых картинок [openimages](https://storage.googleapis.com/openimages/web/index.html)
         
    * [ ] Создание всех возможных (допустимых) искажений (Освещение, обрезка, повороты).
        
        * [x] Смещение с масштабированием с использованием фонового изображения
        * [x] Затемнение, засветление, повороты
        * [x] Наложение полупрозрачных шумовых изображений
        
    * [ ] Создание набора с несколькими обьектами на картинке (для задачи обнаружения)
    * [ ] Отображение набора данных

2) Обучение ИНС

    * [ ] Классификатор
    
        * [ ] На базе готовых
        
            * [ ] Xception
            * [ ] VGG16
            * [ ] VGG19
            * [ ] ResNet, ResNetV2, ResNeXt
            * [ ] InceptionV3
            * [ ] InceptionResNetV2
            * [ ] MobileNet
            * [ ] MobileNetV2
            * [ ] DenseNet
            * [ ] NASNet
            
        * [ ] Собственная разработка
        
    * [ ] Обнаружение и локализация
    
        * [ ] Yolo detetction
        
        * [ ] Обнаружение с использованием фильтров и выделения фигур
        
        * [ ] Распознование на упрощенном ИНС всех знаков, его отсутствия. 
        Возможно оценка по простым признакам группы знаков и его важность. 
        
    * [ ] Перенос на систему реального времени
    
    
#### Enviroment
   
   * HomeBrew - https://brew.sh/index_ru
   * ImageMagic - http://www.imagemagick.org/script/download.php
   * ghostscript - https://github.com/delphian/drupal-convert-file/wiki/Installing-ImageMagick-on-Mac-OSX-for-PHP-and-MAMP
   * Python 3 
   * TensorFlow



### References
* http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
* https://habr.com/company/newprolab/blog/339484/
* http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
* https://habr.com/post/163663/
* https://habr.com/post/142818/
* https://habr.com/company/newprolab/blog/334618/
* Посмотреть https://gym.openai.com
* ГОСТ Р 52290— 2004
* http://www.artpatch.ru/dorznaki.html
* https://habr.com/company/newprolab/blog/334618/
* http://www.chioka.in/class-imbalance-problem/
* http://www.image-net.org
