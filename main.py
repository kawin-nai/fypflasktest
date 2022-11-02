from vgg_utils_withsave import *
from vgg_scratch import *
from flask import Flask
from tensorflow.keras.models import Model
import mtcnn
import os

import cv2

app = Flask(__name__)
img_path = "./content/application_data"
input_path = os.path.join(img_path, "input_faces")
verified_path = os.path.join(img_path, "verified_faces")
vgg_descriptor = None
detector = None


@app.route('/')
def home_endpoint():
    return 'Hello World!'


def load_model():
    global vgg_descriptor, detector
    model = define_model()
    vgg_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    detector = mtcnn.MTCNN()


@app.route('/predict', methods=['GET', 'POST'])
def get_prediction():
    input_img_path = os.path.join(input_path, "input.jpg")

    input_embedding = get_embedding(input_img_path, detector, vgg_descriptor)
    if input_embedding is None:
        raise Exception("No face detected in input image")

    all_distance = {}
    for persons in os.listdir(verified_path):
        # print(persons)
        person_distance = []
        images = []
        for image in os.listdir(os.path.join(verified_path, persons)):
            full_img_path = os.path.join(verified_path, persons, image)
            if full_img_path[-3:] == "jpg":
                images.append(full_img_path)
            # Get embeddings
        embeddings = get_embeddings(images, detector, vgg_descriptor)
        if embeddings is None:
            print("No faces detected")
            continue
        # Check if the input face is a match for the known face
        # print("input_embedding", input_embedding)
        for embedding in embeddings:
            score = is_match(embedding, input_embedding)
            person_distance.append(score)
        # Calculate the average distance for each person
        avg_distance = sum(person_distance) / len(person_distance)
        all_distance[persons] = avg_distance

    # Get the top three persons with the lowest distance
    top_three = sorted(all_distance.items(), key=lambda x: x[1])[:3]
    print(top_three)
    return str(top_three)


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
