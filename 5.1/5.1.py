import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Constants
IMSIZE = 448
BATCH_SIZE = 40
LABELS = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
DATA_DIR = R"C:\Users\rms11\Desktop\Proj\Datasets\flowers"

# Function to create DataFrame
def create_dataframe(dir):
    filepaths, labels = [], []
    for folder in os.listdir(dir):
        folder_path = os.path.join(dir, folder)
        for file in os.listdir(folder_path):
            filepaths.append(os.path.join(folder_path, file))
            labels.append(folder)
    return pd.DataFrame({'filepaths': filepaths, 'labels': labels})

# Data Preparation
df = create_dataframe(DATA_DIR)
train_df, temp_df = train_test_split(df, train_size=0.8, random_state=123, shuffle=True)
valid_df, test_df = train_test_split(temp_df, train_size=0.6, random_state=123, shuffle=True)

# ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, validation_split=0.2)
train_gen = datagen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=(IMSIZE, IMSIZE), class_mode='categorical', batch_size=BATCH_SIZE)
valid_gen = datagen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=(IMSIZE, IMSIZE), class_mode='categorical', batch_size=BATCH_SIZE)
test_gen = datagen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=(IMSIZE, IMSIZE), class_mode='categorical', batch_size=BATCH_SIZE, shuffle=False)

# Model Architecture
base_model = tf.keras.applications.EfficientNetB7(include_top=False, input_shape=(IMSIZE, IMSIZE, 3), pooling='max')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.4),
    layers.Dense(len(LABELS), activation='softmax')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training
history = model.fit(train_gen, epochs=20, validation_data=valid_gen, callbacks=[lr_scheduler, early_stopping])

# Predictions & Evaluation
y_pred = np.argmax(model.predict(test_gen), axis=1)
y_true = test_gen.classes
print(classification_report(y_true, y_pred, target_names=LABELS))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save Model for Mobile Deployment
model.save('5.1.tflite')
print("Model saved as TFLite for Flutter deployment")
