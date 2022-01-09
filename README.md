# darknet-yolov3-object_detection
Обучение нейронной сети yolov3 для детекции объектов трех классов: человек, защитный жилет и каска - с использованием фреймворка darknet.

## pipeline.ipynb
Установка и конфигурирование фреймворка darknet и обучение нейросети yolov3.

## inference.py
Использование обученной нейросети для детекции объектов с использованием OpenCV2.

Сигнатура запуска:

`python inference.py –images_dir images/ -output_dir output/ [-thresh 0.8]`

## Метрики

### во время обучения на тестовом датасете
- раздельные метрики:

Метрика average_precision для класса "protective vest": 92.85%

Метрика average_precision для класса "hard hat": 97.75%

Метрика average_precision для класса "person": 96.59%
- общие метрики:

Метрика precision: 0.90;

Метрика recall: 0.95;

Метрика F1-score: 0.92;

Метрика average IoU: 71.73%;

Метрика mean average precision (mAP): 95.73%.

### на закрытом тестовом датасете
- раздельные метрики:

Метрика average_precision для класса "protective vest": 93.97%

Метрика average_precision для класса "hard hat": 98.09%

Метрика average_precision для класса "person": 96.86%
- общие метрики:

Метрика precision: 0.91;

Метрика recall: 0.96;

Метрика F1-score: 0.93;

Метрика average IoU: 72.91%;

Метрика mean average precision (mAP): 96.31%.
