import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Model, callbacks
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import csv

# === DATA LOADING AND PREPROCESSING ===

dataset_path = R"C:\Users\Rhodri\Desktop\Project\shrooms_ds_validated"
image_size = (224, 224)  # VGG19 input size
batch_size = 32

# Load the dataset
full_dataset = image_dataset_from_directory(
    dataset_path,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
    seed=42
)

# Define dataset sizes for splitting
dataset_size = len(full_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size

# Split the dataset
train_dataset = full_dataset.take(train_size)
remaining_dataset = full_dataset.skip(train_size)
validation_dataset = remaining_dataset.take(val_size)
test_dataset = remaining_dataset.skip(val_size)

# Normalize pixel values
normalization_layer = layers.Rescaling(1.0 / 255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Improved data augmentation
augmentation_layer = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1)
])
train_dataset = train_dataset.map(lambda x, y: (augmentation_layer(x), y))

# Prefetch data
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Extract class names
class_names = full_dataset.class_names
num_classes = len(class_names)
print(f"Class names: {class_names}")

# Compute class weights
true_labels = []
for _, labels in train_dataset.unbatch():  # Fix class weights computation
    true_labels.append(labels.numpy())
true_labels = np.array(true_labels)
class_weights = compute_class_weight('balanced', classes=np.unique(true_labels), y=true_labels)
class_weights = {i: weight for i, weight in enumerate(class_weights)}
print(f"Class weights: {class_weights}")

# === MODEL DEFINITION ===

# Load VGG19
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Unfreeze the last few layers of VGG19 for fine-tuning
for layer in base_model.layers[:-4]:
    layer.trainable = False
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Add custom layers
x = base_model.output
x = BatchNormalization()(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(x)
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(x)
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Complete the model
model = Model(inputs=base_model.input, outputs=predictions)

# === MODEL COMPILATION ===

# Use cosine decay learning rate
initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=10000,
    alpha=0.01
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Create output directory
output_directory = R"C:\Users\Rhodri\Desktop\Project\Results\5.8"
os.makedirs(output_directory, exist_ok=True)

# === CALLBACKS ===
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)

# === MODEL TRAINING ===
epochs = 50
naming_base = f"vgg19_LR{initial_learning_rate}_BS{batch_size}_E{epochs}"
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset,
    class_weight=class_weights,
    callbacks=[reduce_lr, early_stopping]
)

# === TEST-TIME AUGMENTATION (TTA) ===
def test_time_augmentation(model, dataset, augmentations=5):
    predictions = []
    for images, _ in dataset:
        augmented_images = [augmentation_layer(images) for _ in range(augmentations)]
        augmented_preds = [model.predict(aug_img) for aug_img in augmented_images]
        avg_preds = np.mean(augmented_preds, axis=0)  # Average predictions
        predictions.extend(avg_preds)
    return predictions

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

# === SAVE HISTORY ===
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
true_labels = []
predicted_labels = []
for images, labels in test_dataset:
    true_labels.extend(labels.numpy())
    predictions = model.predict(images)
    predicted_labels.extend(np.argmax(predictions, axis=1))

conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_names)))
conf_matrix_display = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names)

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
