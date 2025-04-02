# train_model.py - Trains a CNN model on the collected hand gesture images

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = (64, 64)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

dataset_path = "dataset/"
train_data = datagen.flow_from_directory(dataset_path, target_size=img_size, batch_size=32, class_mode='categorical', subset='training')
val_data = datagen.flow_from_directory(dataset_path, target_size=img_size, batch_size=32, class_mode='categorical', subset='validation')

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=10)
model.save("sign_language_model.h5")