import tensorflow as tf
import matplotlib.pyplot as plt
import csv
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory

# === DATA LOADING AND PREPROCESSING ===

dataset_path = (R"C:\Users\rms11\Desktop\Proj\Datasets\LeafSnap_Leaves_Lab")
image_size = (256, 256)  
batch_size = 32

# Load the dataset and split into training and validation sets
dataset = image_dataset_from_directory(
    dataset_path,
    image_size=image_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=42
)

validation_dataset = image_dataset_from_directory(
    dataset_path,
    image_size=image_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=42
)

# Extract class names automatically from the dataset directory structure
class_names = dataset.class_names
print(f"Class names: {class_names}")

# Normalize pixel values to the range [0, 1] using a Rescaling layer
normalization_layer = layers.Rescaling(1.0 / 255)

# Apply normalization to all datasets
dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# Prefetch data to optimize pipeline performance during training
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

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
              metrics=['accuracy']
              )

history = model.fit(
    dataset,
    epochs=10,
    validation_data=validation_dataset)

# === VISUALIZATION FUNCTION ===

def plot_training_history(history):
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
    plt.show()

# === CSV STORAGE FUNCTION ===

def save_training_history_to_csv(history, filename="training_history.csv"):
    keys = history.history.keys()

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch'] + list(keys))  # Header row
        for epoch in range(len(history.history['accuracy'])):
            row = [epoch + 1] + [history.history[key][epoch] for key in keys]
            writer.writerow(row)

    print(f"Training history saved to {filename}.")


# Visualize training performance
plot_training_history(history)

# Save training metrics to a CSV file
save_training_history_to_csv(history)
