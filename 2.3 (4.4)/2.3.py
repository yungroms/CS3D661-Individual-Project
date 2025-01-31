import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, log_loss

# === DATA LOADING AND PREPROCESSING ===

# Output directory setup
output_directory = R"C:\Users\rms11\Desktop\y3_proj\2.3 (4.4)\2.3_results"
os.makedirs(output_directory, exist_ok=True)

# Path to the saved model
model_path = R"C:\Users\rms11\Desktop\Proj\Best_Models\4.3 (2.2)\1_model_LR0.005_BS32_DR0.5_E15.h5"

# Path to the test dataset directory
test_data_path = R"C:\Users\rms11\Desktop\Proj\Datasets\Web_Dataset"

# Define image size and batch size
image_size = (256, 256)
batch_size = 32

# Extract naming base from input model path
naming_base = os.path.splitext(os.path.basename(model_path))[0]

# === LOAD MODEL AND TEST DATA ===

# Load the saved model
model = load_model(model_path)
print(f"Model loaded from {model_path}.")

# Load the test dataset
test_dataset = image_dataset_from_directory(
    test_data_path,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False  # Maintain order for proper evaluation
)

# Extract class names from the dataset
class_names = test_dataset.class_names
print(f"Class names: {class_names}")

# Normalize the test dataset
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# === TEST THE MODEL ===

# Get true labels, predictions, and probabilities
true_labels = []
predicted_labels = []
probabilities = []
images_to_display = []
for images, labels in test_dataset:
    true_labels.extend(labels.numpy())
    predictions = model.predict(images)
    probabilities.extend(predictions)
    predicted_labels.extend(np.argmax(predictions, axis=1))
    images_to_display.extend(images.numpy())

# === CALCULATE METRICS ===

# Compute accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.4f}")

# Compute log loss
log_loss_value = log_loss(true_labels, probabilities, labels=range(len(class_names)))
print(f"Log Loss: {log_loss_value:.4f}")

# Save accuracy and log loss
metrics_filename = os.path.join(output_directory, f"{naming_base}_metrics.txt")
with open(metrics_filename, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Log Loss: {log_loss_value:.4f}\n")
print(f"Metrics saved to {metrics_filename}.")

# === CONFUSION MATRIX ===

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_names)))
conf_matrix_display = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names)

# Plot and save the confusion matrix
plt.figure(figsize=(10, 8))
conf_matrix_display.plot(cmap='viridis', values_format='d', ax=plt.gca())
plt.title('Confusion Matrix')
plt.xticks(rotation=45)

confusion_matrix_filename = os.path.join(output_directory, f"{naming_base}_confusion_matrix.png")
plt.savefig(confusion_matrix_filename)
plt.show()
print(f"Confusion matrix saved to {confusion_matrix_filename}.")