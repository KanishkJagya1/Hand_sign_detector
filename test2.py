import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define your project's directory structure
project_dir = "./python" # Replace with your project directory
train_dir = os.path.join(project_dir, "./data/C") # Data folder with subdirectories A, B, C
val_dir = os.path.join(project_dir, "./data/C") # You can use the same data folder for validation
model_path = os.path.join(project_dir, "model", "keras_model.h5")

# Parameters
input_shape = (224, 224, 3)  
batch_size = 32
num_classes = 3 # Number of classes (A, B, C)

# Data augmentation
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")

# Load training data
train_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode="categorical")

# Load validation data (You can set up a separate validation dataset if needed)
validation_data_gen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_data_gen.flow_from_directory(
    val_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode="categorical")

# Create and compile the model
base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
x = base_model.output
x = layers.GlobalAveragePooling2D