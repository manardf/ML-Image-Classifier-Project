import sys
import time
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from PIL import Image

BATCH_SIZE = 32
IMAGE_SIZE = 224
class_labels = {}

def preprocess_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image /= 255.0
    return image.numpy()

def make_prediction(image_path, model, top_k=5):
    image = Image.open(image_path)
    image_array = np.asarray(image)
    processed_image = preprocess_image(image_array)
    expanded_image = np.expand_dims(processed_image, axis=0)
    
    predictions = model.predict(expanded_image)[0]
    top_probs = -np.partition(-predictions, top_k)[:top_k]
    top_classes = np.argpartition(-predictions, top_k)[:top_k]
    
    return top_probs, top_classes

if __name__ == '__main__':
    print("Executing predict.py...")

    parser = argparse.ArgumentParser(description="Image Classifier Prediction Script")
    parser.add_argument('image_path', type=str, help="Path to the image file")
    parser.add_argument('--top_k', type=int, help="Number of top predictions to return")
    parser.add_argument('--category_names', type=str, help="Path to JSON file mapping class indices to names")

    args = parser.parse_args()
    print("Arguments received:", args)

    image_path = args.image_path
    top_k = args.top_k if args.top_k else 5

    model = tf.keras.models.load_model('flower_classifier.h5', custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)

    probabilities, class_indices = make_prediction(image_path, model, top_k)

    if args.category_names:
        with open(args.category_names, 'r') as json_file:
            label_map = json.load(json_file)
        class_labels = [label_map.get(str(idx + 1)) for idx in class_indices]

    print("Predicted Probabilities:", probabilities)
    print("Predicted Classes:", class_labels if args.category_names else class_indices)