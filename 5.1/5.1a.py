import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import itertools

# Constants
IMG_SIZE = 448
IMG_CHANNELS = 3
LABELS = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
DATA_DIR = '/kaggle/input/flowers-recognition/flowers/'
BATCH_SIZE = 40
EPOCHS = 20

# Function to define file paths and labels
def define_paths(directory):
    filepaths, labels = [], []
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        for file in os.listdir(folder_path):
            filepaths.append(os.path.join(folder_path, file))
            labels.append(folder)
    return filepaths, labels

# Function to create dataframe
def create_df(directory):
    files, classes = define_paths(directory)
    return pd.DataFrame({'filepaths': files, 'labels': classes})

# Load dataset
df = create_df(DATA_DIR)
train_df, temp_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123)
valid_df, test_df = train_test_split(temp_df, train_size=0.6, shuffle=True, random_state=123)

# Function to create generators
def create_generators(train_df, valid_df, test_df, batch_size):
    img_size = (IMG_SIZE, IMG_SIZE)
    
    def scalar(img):
        return img
    
    tr_gen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True, vertical_flip=True, rotation_range=1)
    ts_gen = ImageDataGenerator(preprocessing_function=scalar)
    
    train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                           class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
    valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                           class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
    test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                          class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)
    return train_gen, valid_gen, test_gen

train_gen, valid_gen, test_gen = create_generators(train_df, valid_df, test_df, BATCH_SIZE)

# Model definition
base_model = tf.keras.applications.EfficientNetB7(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS), pooling='max')
for layer in base_model.layers:
    if layer.name == 'block7d_project_conv':
        break
    layer.trainable = False

x = layers.BatchNormalization()(base_model.output)
x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.016), activity_regularizer=regularizers.l1(0.006),
                 bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
x = layers.Dropout(0.45)(x)
x = layers.Dense(len(train_gen.class_indices), activation='softmax')(x)

model = keras.models.Model(base_model.input, x)
model.compile(optimizer=optimizers.Adamax(learning_rate=0.0011), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(train_gen, epochs=EPOCHS, validation_data=valid_gen, verbose=1)

# Plot Training Results
def plot_training(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

plot_training(history)

# Model Evaluation
y_pred = np.argmax(model.predict(test_gen), axis=1)
cm = confusion_matrix(test_gen.classes, y_pred)
print(classification_report(test_gen.classes, y_pred, target_names=LABELS))

# Confusion Matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

plot_confusion_matrix(cm, LABELS)
