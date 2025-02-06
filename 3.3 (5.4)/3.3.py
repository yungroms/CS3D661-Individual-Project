import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import csv
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import MobileNetV2

# === DATA LOADING AND PREPROCESSING ===

# Specify the path to your leaf dataset directory
dataset_path = R"C:\Users\rms11\Desktop\Proj\Datasets\shrooms_ds_max"
image_size = (224, 224)  # MobileNetV2 input size
batch_size = 32

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

# Normalize pixel values to the range [0, 1]
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

# === MODEL DEFINITION ===

# Load MobileNetV2 with pre-trained weights and exclude the top layers
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')  # Output layer
])

# === MODEL COMPILATION ===

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === MODEL TRAINING ===

learning_rate = 0.0001
epochs = 20
naming_base = f"mobilenet_LR{learning_rate}_BS{batch_size}_E{epochs}"
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset
)

# Create output directory
output_directory = R"C:\Users\rms11\Desktop\y3_proj\3.3 (5.4)\3.3_results"
os.makedirs(output_directory, exist_ok=True)

# === VISUALIZATION FUNCTION ===

def plot_training_history(history, naming_base):
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

    plot_filename = os.path.join(output_directory, f'{naming_base}_training_plot.png')
    plt.savefig(plot_filename)
    plt.show()

# === CSV STORAGE FUNCTION ===

def save_training_history_to_csv(history, naming_base):
    csv_filename = os.path.join(output_directory, f'{naming_base}_training_history.csv')
    keys = history.history.keys()

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch'] + list(keys))
        for epoch in range(len(history.history['accuracy'])):
            row = [epoch + 1] + [history.history[key][epoch] for key in keys]
            writer.writerow(row)

    print(f"Training history saved to {csv_filename}.")

plot_training_history(history, naming_base)
save_training_history_to_csv(history, naming_base)

# === TESTING ===

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save test results
test_results_file = os.path.join(output_directory, f'{naming_base}_test_results.txt')
with open(test_results_file, 'w') as f:
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_accuracy}\n")

print(f"Test results saved to {test_results_file}.")

# === CONFUSION MATRIX ===

# Generate predictions and true labels
true_labels = []
predicted_labels = []
for images, labels in test_dataset:
    true_labels.extend(labels.numpy())
    predictions = model.predict(images)
    predicted_labels.extend(np.argmax(predictions, axis=1))

# Create confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_names)))
conf_matrix_display = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names)

# Plot and save the confusion matrix
plt.figure(figsize=(10, 8))
conf_matrix_display.plot(cmap='viridis', values_format='d', ax=plt.gca())
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
confusion_matrix_filename = os.path.join(output_directory, f'{naming_base}_confusion_matrix.png')
plt.savefig(confusion_matrix_filename)
plt.show()

print(f"Confusion matrix saved to {confusion_matrix_filename}.")

# === SAVE THE MODEL ===

model_filename = os.path.join(output_directory, f"{naming_base}.h5")
model.save(model_filename)
print(f"Model saved to {model_filename}.")