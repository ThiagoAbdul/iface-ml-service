import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("././src/assets/haarcascade_frontalface_alt.xml")


def get_face_image_from_bytes(image_bytes):
    np_arr = np.asarray(image_bytes, dtype=np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.COLOR_BGR2GRAY)
    return get_face_image(cv_image)


def get_face_image(image):
    faces_coordinates = face_cascade.detectMultiScale(image, 1.3, 4)
    face_images = []
    for (x, y, w, h) in faces_coordinates:
        face_image = cv2.resize(image[y:y + h, x:x + w], (28, 28))
        face_images.append(face_image)
    return face_images
