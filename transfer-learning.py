# TensorFlow and tf.keras
import tensorflow as tf
import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import data as tf_data
from keras import layers

image_size = (224, 224)
batch_size = 128
num_classes = 3

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "processed",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

data_augmentation_layers = keras.Sequential([
    keras.layers.RandomFlip("horizontal", input_shape=(224, 224, 3)),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1)
])

# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

model = keras.Sequential([
    data_augmentation_layers,
    layers.Rescaling(1.),
    keras.applications.EfficientNetV2B2(classes=3, include_top=False),
    # layers.Conv2D(32, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=5)

test_loss, test_acc = model.evaluate(val_ds, verbose=2)

print('\nTest accuracy:', test_acc)

# if test_acc > 0.8:
model.save("model")
