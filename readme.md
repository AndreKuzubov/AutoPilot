# Авто Пилот 

Проект к выпускной магисторской работе


## Задачи:
* [ ] [Распознование дорожных знаков](RoadSign)
* [ ] Распознование дорожной разметки
* [ ] Распознование обьектов: Машины, люди, препятствия
* [ ] Распознование номеров машин
* [ ] Распознование звуковых сигналов
* [ ] Распознование жестов патрульного



#### Доп. задачи:
* [ ] Оптимизация работы с использованием видео ряда
* [ ] Тестирование на игровом движке
* [ ] Голосовое управление

## Точки входа

 * [RoadSign/createDataSetOnPatters.py](RoadSign/start_createDataSetOnPatters.py) - генерирование наборов картинок для обучения 
 
 * [RoadSign/explorationDataAnalys.py](RoadSign/explorationDataAnalys.py) - аналитика по сгенерированным картинкам
 
 * [ImageNetModels/start_test_all_models.py](ImageNetModels/start_test_all_models.py) - сравнение предобученных моделей на стандартных картинках
         
 * [RoadSign/start_to_train_sign_recognation.py](RoadSign/start_to_train_sign_recognation.py)
     <b>Запуск:</b>
     
            >> cd <dir>/RoadSign
            >> python start_to_train_sign_recognation.py
            >> tensorboard --logdir=log/tensorboard 
        
     далее пройти по ссылке: [http://localhost:6006](http://localhost:6006) 


## Enviroment
    
  * https://pypi.org/project/transliterate/  
       
## License
```
Copyright (c) 2019 Andrey Kuzubov
```     
