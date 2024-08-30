import cv2
import numpy as np


# extract the face portion in a given image
def extractFace(image, x1, x2, y1, y2):
    image_array = np.asarray(image, "uint8")

    y_min = min(y1, y2)
    y_max = max(y1, y2)
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    face = image_array[y_min:y_max, x_min:x_max]

    # resize the detected face to 224x224: size required for VGGFace input
    try:
        face = cv2.resize(face, (240, 240))
        face_array = np.asarray(face, "uint8")
        return face_array
    except:
        return None


import dlib
import matplotlib.pyplot as plt

detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
conf_thres = 0.6


# detect face in a given image
# returning an array of image_array for all faces detected in a image
# input: image as an array
def detectFace(image):
    image_array = np.asarray(image, "uint8")
    faces_detected = detector(image_array)
    if len(faces_detected) == 0:
        return []
    faces_extracted = []

    for face in faces_detected:

        conf = face.confidence
        if conf < conf_thres:
            continue

        x1 = face.rect.left()
        y1 = face.rect.bottom()
        x2 = face.rect.right()
        y2 = face.rect.top()

        face_array = extractFace(image, x1, x2, y1, y2)
        if face_array is not None:
            faces_extracted.append(face_array)

    return faces_extracted

import os

import cv2
# from image_processing import load_face

def load_face(dir):
    faces = []
    # enumerate files
    for filename in os.listdir(dir):
        if filename.endswith("png") or filename.endswith("jpg") or filename.endswith("jpeg"):
            path = os.path.join(dir, filename)
            image = cv2.imread(path)
            face_array = detectFace(image)
            faces.extend(face_array)

    return faces


original_folder_name = 'original'
original_images_dir = os.path.join(".", original_folder_name)

labels = []
faces_data = []

for dirName in os.listdir(original_images_dir):
    path = os.path.join(original_images_dir, dirName)
    if os.path.isdir(path):
        faces = load_face(path)
        faces_data.append(faces)
        labels.append(dirName)

from PIL import Image

processed_folder_name = 'processed'
processed_folder_dir = os.path.join(".", processed_folder_name)

for index, faces in enumerate(faces_data):
    l = labels[index]
    dir = os.path.join(processed_folder_dir, l)
    if not os.path.exists(dir):
        os.makedirs(dir)
    for index2, face in enumerate(faces):
        im = Image.fromarray(face)
        im.save(os.path.join(dir, f'{index2}.jpg' ))

from PIL import Image

processed_folder_name = 'processed'
processed_folder_dir = os.path.join(".", processed_folder_name)

for index, faces in enumerate(faces_data):
    l = labels[index]
    dir = os.path.join(processed_folder_dir, l)
    if not os.path.exists(dir):
        os.makedirs(dir)
    for index2, face in enumerate(faces):
        im = Image.fromarray(face)
        im.save(os.path.join(dir, f'{index2}.jpg' ))