from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model1.h5")

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust size to match your model input
    image = np.array(image) / 255.0   # Normalize if needed
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(file.stream)
    processed_image = preprocess_image(image)
    
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])  # Assuming classification model
    
    return jsonify({"predicted_class": int(predicted_class)})

if __name__ == "__main__":
    app.run(debug=True)
