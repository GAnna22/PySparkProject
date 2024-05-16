# import findspark
# findspark.init()
#
# from pyspark.ml.classification import MultilayerPerceptronClassifier
# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from pyspark.sql import SparkSession
# import tensorflow as tf
#
# # Создать SparkSession
# spark = SparkSession.builder.appName("MNIST").getOrCreate()
#
# # Загружаем датасет MNIST из интернета с использованием TensorFlow
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# # Преобразуем данные в формат, подходящий для Spark DataFrame
# train_data = [(x.flatten().tolist(), float(y)) for x, y in zip(x_train, y_train)]
# test_data = [(x.flatten().tolist(), float(y)) for x, y in zip(x_test, y_test)]
#
# train_df = spark.createDataFrame(train_data, ["features", "label"])
# test_df = spark.createDataFrame(test_data, ["features", "label"])
#
# # # Преобразовать данные в формат, пригодный для обучения
# # assembler = VectorAssembler(inputCols=["features"], outputCol="features_vector") # VectorAssembler(inputCols=mnist_df.columns[1:], outputCol="features")
# # mnist_assembled_df = assembler.transform(mnist_df)
# #
# # # Разделить данные на обучающий и тестовый наборы
# # (training_df, test_df) = mnist_assembled_df.randomSplit([0.8, 0.2])
#
# # Обучить модель нейронной сети
# nn = MultilayerPerceptronClassifier(maxIter=100, layers=[784, 128, 64, 10])
# model = nn.fit(train_df)
#
# # Оценить модель на тестовом наборе
# predictions = model.transform(test_df)
# evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
# accuracy = evaluator.evaluate(predictions)
#
# # Вывести точность
# print(f"Точность: {accuracy}")
#
# # Сохранить обученную модель
# model.save("data/mnist_model")
# spark.stop()


from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import tensorflow as tf

# Создаем сессию Spark
spark = SparkSession.builder.appName("MNISTClassifier").getOrCreate()
#
# # Загружаем датасет MNIST из TensorFlow
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# # Преобразуем данные в формат, подходящий для Spark DataFrame
# train_data = [(x.flatten().tolist(), float(y)) for x, y in zip(x_train, y_train)]
# test_data = [(x.flatten().tolist(), float(y)) for x, y in zip(x_test, y_test)]
#
# train_df = spark.createDataFrame(train_data, ["features", "label"])
# test_df = spark.createDataFrame(test_data, ["features", "label"])
#
# # Создаем вектор признаков
# assembler = VectorAssembler(inputCols=["features"], outputCol="features_vector")
#
# # Инициализируем модель случайного леса
# rf = RandomForestClassifier(labelCol="label", featuresCol="features_vector", numTrees=10)
#
# # Создаем Pipeline для объединения шагов обработки данных и моделирования
# pipeline = Pipeline(stages=[assembler, rf])
#
# # Обучаем модель на тренировочных данных
# model = pipeline.fit(train_df)
#
# # Прогнозируем метки на тестовых данных
# predictions = model.transform(test_df)
#
# # Оцениваем качество модели
# evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
# accuracy = evaluator.evaluate(predictions)
# print("Accuracy:", accuracy)
#
# # Вычисляем дополнительные метрики
# true_positive = predictions.filter("prediction = label").count()
# false_positive = predictions.filter("prediction != label").count()
# precision = true_positive / (true_positive + false_positive)
# recall = true_positive / test_df.count()
# f1_score = 2 * (precision * recall) / (precision + recall)
#
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1_score)
#
# # Закрываем сессию Spark
# spark.stop()
df_training = (spark
               .read
               .options(header = True, inferSchema = True)
               .csv("data/MNIST/mnist_train.csv"))

print(df_training.count())
feature_columns = df_training.columns[1:]

from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import pandas as pd

vectorizer = VectorAssembler(inputCols=feature_columns, outputCol="features")
training = (vectorizer
            .transform(df_training)
            .select("label", "features")
            .toDF("label", "features")
            .cache())
training.show()

a = training.first().features.toArray()
plt.imshow(a.reshape(28, 28), cmap="Greys")

images = training.sample(False, 0.01, 1).take(25)
fig, _ = plt.subplots(5, 5, figsize = (10, 10))
for i, ax in enumerate(fig.axes):
    r = images[i]
    label = r.label
    features = r.features
    ax.imshow(features.toArray().reshape(28, 28), cmap = "Greys")
    ax.set_title("True: " + str(label))

plt.tight_layout()
