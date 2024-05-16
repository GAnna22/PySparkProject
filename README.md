Чтобы запустить проект, нужно:
1) Создать и подключиться к виртуальному окружению:
    - python3 -m venv venv
    - source venv/bin/activate
    - pip install -r requirements.txt
2) Запустить Kafka на localhost:29093 (docker-compose -f ./docker-compose.yml up -d)
2) Запустить Consumer и распознаватель на pyspark (python3 PysparkConsumer.py)
3) Запустить user interface (streamlit run UI.py)

Чтобы переобучить модель или создать новую => TrainModelMNIST.ipynb
(чтобы подключить ядро с библиотеками из venv, нужно выполнить команду:
python3 -m ipykernel install --user --name venv --display-name "Python (pyspark_env)")
