from confluent_kafka import Producer
import os

image_folder = "data/digits"

producer = Producer({'bootstrap.servers': 'localhost:29093'})

image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

for image_file in image_files:
    with open(os.path.join(image_folder, image_file), 'rb') as file:
        image_data = file.read()
        producer.produce('image_topic', key=image_file.encode(), value=image_data)
        producer.flush()
        print(f"Sent image {image_file} to Kafka")
