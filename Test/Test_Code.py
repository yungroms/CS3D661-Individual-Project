import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.utils import image_dataset_from_directory
from PIL import Image

# === PATHS ===
TEST_DIR = R"C:\Users\rms11\Desktop\y3_proj\Test\Test_Dataset"
OUTPUT_DIR = R"C:\Users\rms11\Desktop\y3_proj\Test\Results"
MODEL_PATH = R"C:\Users\rms11\Desktop\y3_proj\App\classifier_app\model\model2.h5"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# === LOAD MODEL ===
model = tf.keras.models.load_model(MODEL_PATH)

# === LOAD TEST DATASET WITHOUT PREFETCHING/MAPPING FIRST ===
raw_test_dataset = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# === SAVE CLASS NAMES BEFORE MAPPING ===
class_names = raw_test_dataset.class_names
print(f"Class names: {class_names}")

# === APPLY PREPROCESSING (Rescaling like training) ===
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
test_dataset = raw_test_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# === PREDICTIONS ===
true_labels = []
predicted_labels = []
image_samples = []  # To store sample images for visualization
true_sample_labels = []  # Store true labels for sample images
predicted_sample_labels = []  # Store predicted labels for sample images

for images, labels in test_dataset:
    true_labels.extend(labels.numpy())
    predictions = model.predict(images)
    predicted_labels.extend(np.argmax(predictions, axis=1))
    
    # Select 9 random images for visualization
    if len(image_samples) < 9:
        random_indices = np.random.choice(len(images), size=9, replace=False)
        image_samples.extend([images[i] for i in random_indices])
        true_sample_labels.extend([labels[i] for i in random_indices])
        predicted_sample_labels.extend([np.argmax(predictions[i]) for i in random_indices])

# === CONFUSION MATRIX ===
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_names)))
conf_matrix_display = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names)

plt.figure(figsize=(10, 8))
conf_matrix_display.plot(cmap='viridis', values_format='d', ax=plt.gca())
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
confusion_matrix_filename = os.path.join(OUTPUT_DIR, f"mobilenet_test_confusion_matrix.png")
plt.savefig(confusion_matrix_filename)
plt.show()

# === CLASSIFICATION REPORT ===
report = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)
report_csv = os.path.join(OUTPUT_DIR, f"mobilenet_test_classification_report.csv")
with open(report_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    for class_label, metrics in report.items():
        if isinstance(metrics, dict):
            writer.writerow([class_label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])
print(f"Classification report saved to {report_csv}.")

# === TEST LOSS & ACCURACY ===
test_loss, test_accuracy = model.evaluate(test_dataset)
with open(os.path.join(OUTPUT_DIR, 'mobilenet_test_results.txt'), 'w') as f:
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_accuracy}\n")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# === DISPLAY TILE OF 9 RANDOM IMAGES ===
# Convert tensor images to numpy arrays and display them with labels
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()

for i in range(9):
    ax = axes[i]
    ax.imshow(image_samples[i].numpy())  # Convert tensor to numpy for displaying
    ax.set_title(f"True: {class_names[true_sample_labels[i]]}\nPred: {class_names[predicted_sample_labels[i]]}")
    ax.axis('off')

tile_filename = os.path.join(OUTPUT_DIR, 'random_image_tile.png')
plt.tight_layout()
plt.savefig(tile_filename)
plt.show()

