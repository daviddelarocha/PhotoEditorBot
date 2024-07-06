import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

class TrainingProgress(Callback):
    def __init__(self, progress_var, progress_label, epochs):
        super().__init__()
        self.progress_var = progress_var
        self.progress_label = progress_label
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        progress = ((epoch + 1) / self.epochs) * 100
        self.progress_var.set(progress)
        self.progress_label.config(text=f"Progress: {int(progress)}%")

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [1080, 1080])
    image = image / 255.0
    return image

def create_dataset(original_dir, edited_dir, batch_size, val_split=0.2):
    original_images = sorted([os.path.join(original_dir, img) for img in os.listdir(original_dir)])
    edited_images = sorted([os.path.join(edited_dir, img) for img in os.listdir(edited_dir)])

    original_train, original_val, edited_train, edited_val = train_test_split(
        original_images, edited_images, test_size=val_split, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((original_train, edited_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((original_val, edited_val))

    def load_and_pair_images(original_path, edited_path):
        original_image = load_image(original_path)
        edited_image = load_image(edited_path)
        return original_image, edited_image

    train_dataset = train_dataset.map(load_and_pair_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.map(load_and_pair_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset

def train_model(data_path, output_model_path, progress_var, progress_label, batch_size=16, epochs=15):
    train_dataset, val_dataset = create_dataset(
        os.path.join(data_path, 'original'), os.path.join(data_path, 'edited'), batch_size)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(1080, 1080, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(3)  # Assuming the output is an RGB image
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(output_model_path, save_best_only=True, monitor='val_loss', mode='min')
    progress_callback = TrainingProgress(progress_var, progress_label, epochs)

    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[checkpoint, progress_callback])

def process_image(image_path, model_path, output_image_path):
    model = tf.keras.models.load_model(model_path)
    image = load_image(image_path)
    image_expanded = tf.expand_dims(image, axis=0)
    
    predicted_image = model.predict(image_expanded)
    predicted_image = tf.squeeze(predicted_image, axis=0)
    predicted_image = tf.clip_by_value(predicted_image, 0.0, 1.0)
    predicted_image = tf.image.convert_image_dtype(predicted_image, dtype=tf.uint8)

    encoded_image = tf.io.encode_jpeg(predicted_image)
    tf.io.write_file(output_image_path, encoded_image)
