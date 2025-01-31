import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow as tf
import csv
import os

# === DATA LOADING AND PREPROCESSING ===

# Specify the path to your leaf dataset directory
dataset_path = R"C:\Users\rms11\Desktop\Proj\4.0_LeafSnap_Merged\LeafSnap_Leaves_Merged"

# Define the target image size and batch size
image_size = (256, 256)  # Resize all images to 256x256 to match model input requirements
batch_size = 32  # Number of images processed in each batch

# Load the entire dataset without validation split
full_dataset = image_dataset_from_directory(
    dataset_path,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
    seed=42
)

# Define dataset sizes for splitting (70% training, 20% validation, 10% test)
dataset_size = len(full_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size

# Split the dataset
train_dataset = full_dataset.take(train_size)
remaining_dataset = full_dataset.skip(train_size)
validation_dataset = remaining_dataset.take(val_size)
test_dataset = remaining_dataset.skip(val_size)

# Normalize pixel values to the range [0, 1] using a Rescaling layer
normalization_layer = layers.Rescaling(1.0 / 255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Prefetch data to optimize pipeline performance during training
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Extract class names automatically from the dataset directory structure
class_names = full_dataset.class_names
print(f"Class names: {class_names}")

# === MODEL ARCHITECTURE ===

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

# === MODEL COMPILATION ===

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Create output directory
output_directory = R"C:\Users\rms11\Desktop\Proj\4.0_LeafSnap_Merged\Experiment_Results_4.1"
os.makedirs(output_directory, exist_ok=True)

# === VISUALIZATION FUNCTION ===
def plot_training_history(history, learning_rate, batch_size, dropout_rate, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plot_filename = os.path.join(output_directory, f'training_plot_LR{learning_rate}_BS{batch_size}_DR{dropout_rate}_E{epochs}.png')
    plt.savefig(plot_filename)
    plt.show()

# === CSV STORAGE FUNCTION ===
def save_training_history_to_csv(history, learning_rate, batch_size, dropout_rate, epochs):
    csv_filename = os.path.join(output_directory, f'training_history_LR{learning_rate}_BS{batch_size}_DR{dropout_rate}_E{epochs}.csv')
    keys = history.history.keys()

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch'] + list(keys))
        for epoch in range(len(history.history['accuracy'])):
            row = [epoch + 1] + [history.history[key][epoch] for key in keys]
            writer.writerow(row)

    print(f"Training history saved to {csv_filename}.")

# === MODEL TRAINING ===
learning_rate = 0.0005
epochs = 20
dropout_rate = 0.5
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset
)

plot_training_history(history, learning_rate, batch_size, dropout_rate, epochs)
save_training_history_to_csv(history, learning_rate, batch_size, dropout_rate, epochs)

# === TESTING ===
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save test results
test_results_file = os.path.join(output_directory, 'test_results.txt')
with open(test_results_file, 'w') as f:
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_accuracy}\n")

print(f"Test results saved to {test_results_file}.")