**Требования к реализации проекта**:

1. Исходные коды проекта должны находиться в репозитории GitHub.
  
2. Проект оркестируется с помощью ci/cd (jenkins или gitlab).

3. Датасеты версионируются с помощью dvc и синхронизируются с удалённым хранилищем.

4. Разработка возможностей приложения должна проводиться в отдельных ветках, наборы фичей и версии данных тоже.

5. В коневеере запускаются не только модульные тесты, но и проверка тестами на качество данных.

6. Итоговое приложение реализуется в виде образа docker. Сборка образа происходит в конвеере.

7. В проекте может использоваться предварительно обученная модель. Обучать собственную модель не требуется.

В ходе выполнения проекта была произведена проверка тестов на Jenkins. Ниже приведен скриншот, иллюстрирующий стадии выполнения конвейера и время, затраченное на каждую из них.

![Описание изображения](images/Jenkins.jpg)
