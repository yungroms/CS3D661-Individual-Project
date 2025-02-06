import csv
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
# Paths and dataset setup
dataset_dir = pathlib.Path(R"C:\Users\rms11\Desktop\Proj\Datasets\shrooms_ds")
img_height, img_width = 224, 224
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)

# Hyperparameters + Naming Convention
architecture_name = "EfficientNetV2S"
learning_rate = 0.001
epochs = 10
output_name = f"{architecture_name}_lr{learning_rate}_epochs{epochs}"

# Model definition
full_model = Sequential()
pretrained_model = EfficientNetV2S(
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg',
    classes=num_classes,
    weights='imagenet'
)

for layer in pretrained_model.layers:
    layer.trainable = False

full_model.add(pretrained_model)
full_model.add(Flatten())
full_model.add(Dense(512, activation='relu'))
full_model.add(Dense(num_classes, activation='softmax'))

full_model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback to save training/validation metrics to CSV
class CSVLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, file_name):
        self.file_name = file_name
        with open(self.file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])

    def on_epoch_end(self, epoch, logs=None):
        with open(self.file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, logs['loss'], logs['accuracy'], logs['val_loss'], logs['val_accuracy']])

metrics_csv = f"{output_name}_metrics.csv"
csv_logger = CSVLoggerCallback(metrics_csv)

# Model training
history = full_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[csv_logger]
)

# Save model
model_path = f"{output_name}_model.h5"
full_model.save(model_path)

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(f"{output_name}_plots.png")
plt.show()

# Test function
def evaluate_model(model, test_data, output_name):
    # Evaluate on test data
    results = model.evaluate(test_data)
    test_metrics_csv = f"{output_name}_test_metrics.csv"
    with open(test_metrics_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['loss', 'accuracy'])
        writer.writerow(results)

    # Confusion matrix
    y_true = np.concatenate([y for x, y in test_data], axis=0)
    y_pred = np.argmax(model.predict(test_data), axis=-1)
    y_true = np.argmax(y_true, axis=-1)

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f"{output_name}_confusion_matrix.png")
    plt.show()

    print(classification_report(y_true, y_pred, target_names=class_names))

# Load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir.parent / 'test',  # Targeting the test set from inside the same directory as the training set
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

# Evaluate model
evaluate_model(full_model, test_ds, output_name)
