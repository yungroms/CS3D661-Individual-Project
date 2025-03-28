import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy

# Paths and dataset setup
dataset_dir = pathlib.Path(r"C:\Users\rms11\Desktop\Proj\Datasets\shrooms_ds_max_split\train")
output_dir = pathlib.Path(r"C:\Users\rms11\Desktop\y3_proj\4.4 (6.14)\4.4_results")
os.makedirs(output_dir, exist_ok=True)

img_height, img_width = 224, 224
batch_size = 32

# Enhanced data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomShear(0.2)
])

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size
).map(lambda x, y: (data_augmentation(x), y))  # Apply data augmentation to training data

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = sorted([item.name for item in dataset_dir.glob("*") if item.is_dir()])
num_classes = len(class_names)

# Calculate class weights
print("Calculating class weights...")
class_labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
class_indices = np.arange(len(class_names))
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=class_indices,
    y=np.argmax(class_labels, axis=1)
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Class weights:", class_weight_dict)

# Model definition
full_model = Sequential()
pretrained_model = EfficientNetV2S(
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg',
    weights='imagenet'
)

# Freeze all layers initially
for layer in pretrained_model.layers:
    layer.trainable = False

# Add fine-tuning by unfreezing the last 40 layers
for layer in pretrained_model.layers[-40:]:
    layer.trainable = True

full_model.add(pretrained_model)
full_model.add(Flatten())
full_model.add(Dense(512, activation='relu'))
full_model.add(Dense(num_classes, activation='softmax'))

# Compile model with Categorical Crossentropy
initial_learning_rate = 0.001
full_model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Callbacks
class CSVLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, file_path):
        self.file_path = file_path
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])

    def on_epoch_end(self, epoch, logs=None):
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, logs['loss'], logs['accuracy'], logs['val_loss'], logs['val_accuracy']])

metrics_csv = output_dir / "metrics.csv"
csv_logger = CSVLoggerCallback(metrics_csv)

# Learning rate scheduler
def lr_scheduler(epoch):
    return initial_learning_rate * 0.1 ** (epoch // 5)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

# Training
print("Training model...")
history = full_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,  # Increased epochs for better training
    callbacks=[csv_logger, lr_schedule, early_stopping],
    class_weight=class_weight_dict  # Include class weights
)

# Save model
model_path = output_dir / "trained_model.h5"
full_model.save(model_path)

# Plot accuracy and loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plot_path = output_dir / "training_plots.png"
plt.savefig(plot_path)
plt.show()

# Test function
def evaluate_model(model, test_data, output_name, output_dir):
    # Evaluate on test data
    results = model.evaluate(test_data)
    test_metrics_csv = output_dir / f"{output_name}_test_metrics.csv"
    with open(test_metrics_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['loss', 'accuracy'])
        writer.writerow(results)

    # Confusion matrix and classification report
    y_true = np.concatenate([y for x, y in test_data], axis=0)
    y_pred = np.argmax(model.predict(test_data), axis=-1)
    y_true = np.argmax(y_true, axis=-1)

    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')  # Normalized confusion matrix
    conf_matrix_csv = output_dir / f"{output_name}_confusion_matrix.csv"
    np.savetxt(conf_matrix_csv, conf_matrix, delimiter=",", fmt='0.2f')

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_plot_path = output_dir / f"{output_name}_confusion_matrix.png"
    plt.savefig(cm_plot_path)
    plt.show()

    # Generate classification report
    try:
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_csv = output_dir / f"{output_name}_classification_report.csv"
        with open(report_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
            for class_label, metrics in report.items():
                if isinstance(metrics, dict):  # Skip "accuracy" or other summary keys
                    writer.writerow([class_label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])
    except Exception as e:
        print(f"Error generating classification report: {e}")

# Load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir.parent / 'test',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

# Evaluate model
evaluate_model(full_model, test_ds, "final_model", output_dir)
