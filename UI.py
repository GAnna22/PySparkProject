import streamlit as st
from confluent_kafka import Producer, Consumer
import time
from PIL import Image
import io

st.title("Распознавание цифр на фото с использованием Pyspark и Kafka")

@st.cache_resource
def get_producer():
    prod = Producer({'bootstrap.servers': 'localhost:29093'})
    return prod

@st.cache_resource
def get_consumer():
    cons = Consumer({
        'bootstrap.servers': 'localhost:29093',
        'group.id': 'image-consumer',
        'auto.offset.reset': 'latest'
    })
    cons.subscribe(['digit_topic2'])
    return cons


producer = get_producer()
consumer = get_consumer()

uploaded_file = st.file_uploader("Choose photo")
if uploaded_file is not None:
    filename = uploaded_file.name
    bytes_data = uploaded_file.getvalue()

    with st.spinner(f'Sending image {filename} to Kafka...'):
        producer.produce('image_topic2', key=filename.encode(), value=bytes_data)
        producer.flush()
    st.success(f"Sent image {filename} to Kafka")

    waiting_time = 15
    latest_time = time.time()
    img = Image.open(io.BytesIO(bytes_data))
    st.image(img.resize((200, 200)))
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            if time.time() - latest_time > waiting_time:
                st.info("No results. Closing consumer...")
                break
            continue
        elif msg.error():
            st.info("Consumer error: {}".format(msg.error()))
            continue
        else:
            name = msg.key().decode('utf-8')
            if filename == name:
                st.write((f"Recognition result for {name} "
                          f"is **{msg.value().decode('utf-8')}**"))
                break
