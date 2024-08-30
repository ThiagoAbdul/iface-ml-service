import os

import cv2
from image_processing import load_face

from resize import detectFace


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
        label.append(dirName)