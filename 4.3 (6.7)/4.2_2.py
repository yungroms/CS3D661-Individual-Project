import csv
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import collections

# Paths and dataset setup
dataset_dir = (R"C:\Users\rms11\Desktop\Proj\Datasets\LeafSnap_15_merged_split\train")
output_dir = pathlib.Path(r"C:\Users\rms11\Desktop\y3_proj\4.3 (6.7)\4.3_results")
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

# Naming convention
architecture_name = "EfficientNetB7"
learning_rate = 0.001
epochs = 10
output_name = f"{architecture_name}_lr{learning_rate}_epochs{epochs}_bs{batch_size}"

# Model definition
full_model = Sequential()
pretrained_model = EfficientNetB7(
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

    # Manual accuracy check
    manual_accuracy = np.mean(y_pred == y_true)
    print(f"Manual Accuracy Check: {manual_accuracy}")

    # Check label consistency
    print("Class Names:", class_names)
    print("Sample True Labels:", y_true[:10])
    print("Sample Predicted Labels:", y_pred[:10])

    # Check for class imbalance
    class_counts = collections.Counter(y_pred)
    print("Predicted Class Distribution:", class_counts)

    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_csv = output_dir / f"{output_name}_confusion_matrix.csv"
    np.savetxt(conf_matrix_csv, conf_matrix, delimiter=",", fmt='%d')

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_plot_path = output_dir / f"{output_name}_confusion_matrix.png"
    plt.savefig(cm_plot_path)
    plt.show()

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_csv = output_dir / f"{output_name}_classification_report.csv"
    with open(report_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        for class_label, metrics in report.items():
            if isinstance(metrics, dict):
                writer.writerow([class_label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])

# Load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir.parent / 'test',  # Assuming test dataset is in a sibling directory to 'train'
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

# Evaluate model
evaluate_model(full_model, test_ds, output_name, output_dir)
