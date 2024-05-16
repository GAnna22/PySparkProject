#!/usr/bin/env python

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
from confluent_kafka import Consumer, Producer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import io
from PIL import Image

def display_image(image_data):
    img = mpimg.imread(image_data, format='jpg')
    plt.imshow(img)
    plt.show()

def define_digit(lr_model, spark_session, image_data):
    image = np.array(Image.open(image_data)).flatten()
    feature_columns = ["c_" + str(i) for i in range(len(image))]
    df = pd.DataFrame([image], columns=feature_columns)
    image_spark_df = spark_session.createDataFrame(df)
    vectorizer = VectorAssembler(inputCols=feature_columns, outputCol="features")
    image_spark_df_test = (vectorizer
                           .transform(image_spark_df)
                           .select("features")
                           .toDF("features")
                           .cache())
    image_test_pred = lr_model.transform(image_spark_df_test)
    recogn_value = image_test_pred.select("prediction").collect()[0].prediction
    return recogn_value

def main():
    spark = SparkSession.builder.appName("KafkaToImage").getOrCreate()
    lr_model = LogisticRegressionModel.load("lr_model")

    consumer = Consumer({
        'bootstrap.servers': 'localhost:29093',
        'group.id': 'image-consumer',
        'auto.offset.reset': 'latest'
    })
    consumer.subscribe(['image_topic2'])

    producer = Producer({'bootstrap.servers': 'localhost:29093'})

    try:
        while True:
            msg = consumer.poll(2.0)
            if msg is None:
                print('Waiting...')
                continue
            if msg.error():
                print("Consumer error: {}".format(msg.error()))
                continue

            image_data = io.BytesIO(msg.value())
            # display_image(image_data)
            recogn_value = define_digit(lr_model, spark, image_data)
            print(f"{msg.key().decode('utf-8')}: {recogn_value}")
            producer.produce('digit_topic2', key=msg.key(), value=str(recogn_value).encode())
            producer.flush()
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        spark.stop()

if __name__ == '__main__':
    main()
