import face_recognition as fr
import cv2
import numpy as np

img1 = cv2.cvtColor(fr.load_image_file("./src/assets/img1.jpeg"), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(fr.load_image_file("./src/assets/img3.jpg"), cv2.COLOR_BGR2RGB)
encode_img1 = fr.face_encodings(img1)[0]
encode_img2 = fr.face_encodings(img2)[0]

cmp = fr.compare_faces([encode_img1], encode_img2)


def _every(*assertions: bool):
    assertions[0]


def compare_images(img):
    encode_img = fr.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0]
    return all(fr.compare_faces([encode_img1, encode_img2], encode_img))

# cv2.imshow("billie", img1)
# cv2.waitKey(0)
