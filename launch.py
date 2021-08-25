# import gradio

# def hello(inp):
#   return "Hello!"

# image = gradio.inputs.Image(label="Input image", source="webcam")

# io = gradio.Interface(fn=hello, live=True, inputs=image, outputs='text', title='Hello World', 
#     description='The simplest Hosted interface.')

# io.launch()

# Demo: (Image) -> (Label)

import gradio as gr
import tensorflow as tf
import numpy as np
import json
from os.path import dirname, realpath, join

# Load human-readable labels for ImageNet.
current_dir = dirname(realpath(__file__))
with open(join(current_dir, "files/imagenet_labels.json")) as labels_file:
    labels = json.load(labels_file)

mobile_net = tf.keras.applications.MobileNetV2()
def image_classifier(im):
    arr = np.expand_dims(im, axis=0)
    arr = tf.keras.applications.mobilenet.preprocess_input(arr)
    prediction = mobile_net.predict(arr).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}

iface = gr.Interface(
    image_classifier, 
    gr.inputs.Image(shape=(224, 224)), 
    gr.outputs.Label(num_top_classes=3),
    capture_session=True,
    interpretation="default",
    examples=[
        ["images/cheetah1.jpg"],
        ["images/lion.jpg"]
    ])

iface.launch()
