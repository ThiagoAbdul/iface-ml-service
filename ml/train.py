import gc
import numpy as np
from sklearn.model_selection import train_test_split
from face_detect import get_face_image_from_bytes
from keras import callbacks, utils, applications, optimizers, Sequential, Layer, layers, losses, datasets
import tensorflow as tf
import pathlib
from PIL import Image


# def get_image_as_bytes(img_path: str):
#     # f = open(img_path, "rb")
#     # r = bytearray(f.read())
#     # f.close()
#     # return r
#     return Image.open(img_path)
epoch = 300
batch_size = 64
img_height = 28
img_width = 28
data_dir = pathlib.Path("dataset").with_suffix('')



train_ds = utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# # image_array = []
# # label_array = []
# #
# # for i in range(3):
# #     path = base_path + str(i + 1) + ".jpg"
# #     image = get_image_as_bytes(path)
# #     for face_image in get_face_image_from_bytes(image):
# #         image_array.append(face_image)
# #         label_array.append("billie")
# #
# # gc.collect()
#
# # for image in image_array:
# #     cv2.imshow("", image)
# #     cv2.waitKey(500)
#
# # image_np_array = np.array(image_array) / 255.0
# # label_np_array = np.array(label_array)
# # images_train, images_test, labels_train, labels_test = train_test_split(image_np_array,
# #                                                                         label_np_array,
# #                                                                         test_size=0.2)
# #
# # train_dataset = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
# # test_dataset = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
# #
# # mnist = datasets.mnist
#
# # (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # x_train, x_test = x_train / 255.0, x_test / 255.0
# # print(np.shape(x_train))
# # print(np.shape(image_array))
#
# # pretrained_model = applications.MobileNetV2(input_shape=(192, 192, 3),
# #                                             include_top=False,
# #                                             weights="imagenet")
# model = Sequential([
#     # pretrained_model,
#     layers.Flatten(input_shape=(28, 28)),
#     # layers.GlobalAveragePooling2D(),
#     # layers.Dropout(0.2),
#     layers.Dense(128, activation="relu"),
#     layers.Dense(10)
# ])
# #
# # # model.add(pretrained_model.layers)
# #
# model.compile(
#     optimizer='adam',
#     loss=losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=['mae'])
# #
# model_checkpoint = callbacks.ModelCheckpoint(filepath="./src/assets/models/model.weights.h5",
#                                              monitor="val_mae", mode="auto",
#                                              save_best_only=True,
#                                              save_weights_only=True)
#
# reduce_lr = callbacks.ReduceLROnPlateau(factor=0.9,
#                                         monitor="val_mae",
#                                         mode="auto", cooldown=0,
#                                         patience=5,
#                                         verbose=1,
#                                         min_lr=1e-6)
#
#
# # history = model.fit(train_dataset,
# #                     validation_data=test_dataset,
# #                     epochs=epoch,
# #                     batch_size=batch_size,
# #                     callbacks=[model_checkpoint, reduce_lr])
