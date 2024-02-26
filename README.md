# Отчет
### Датасет и обработка данных
Датасет взят на Kaggle.com
https://www.kaggle.com/datasets/pierremegret/dialogue-lines-of-the-simpsons
Он представляет собой диалоги из мультфильма Симпсоны. 
Обработка данных выполнена в ноутбуке [HW_1_bi_encoder_homer.ipynb](https://github.com/maxbobrov85/chat_bot/blob/main/HW_1_bi_encoder_homer.ipynb). Из исходных данных были отобраны диалоги с участием Гомера и Лизы. Далее они были разбиты на вопрос/ответ с контекстом. После обработки сформирован окончательный датасет [final_dataset_homer.csv](https://github.com/maxbobrov85/chat_bot/blob/main/final_dataset_homer.csv)
### BI энкодер
В этом же ноутбуке обучили BI энкодер на основе модели 'distilbert-base-uncased'. Обучение производилось на платформе Google Colab (GPU T4). График обучения модели (training_loss) представлен на рисунке
![Без имени](https://github.com/maxbobrov85/chat_bot/assets/114837957/7bf08a65-16a0-4a7f-b94c-2042c7d53489)
### Cross энкодер
Обучение кросс энкодера и формирование окончательной модели описано в ноутбуке [HW_1_cross_encoder_homer.ipynb](https://github.com/maxbobrov85/chat_bot/blob/main/HW_1_cross_encoder_homer.ipynb). График обучения модели (training_loss) представлен на рисунке
![Без имени](https://github.com/maxbobrov85/chat_bot/assets/114837957/ed01fad8-63fb-4ab7-9491-ec86a1c4940f)
Модели [bi-encoder](https://github.com/maxbobrov85/chat_bot/blob/main/bi_encoder_homer) и cross-encoder были сохранены для передачи в приложение flask
### Инференс
