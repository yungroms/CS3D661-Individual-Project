import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = r"static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure directory exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Load TensorFlow model
MODEL_PATH = r"model/model2.h5"
MODEL = tf.keras.models.load_model(MODEL_PATH)

# Define class names (match your training dataset's class names)
class_names = ["Bluebell", "Buttercup", "Coltsfoot", "Cowslip", "Crocus",
               "Daffodil", "Daisy", "Dandelion", "Fritillary", "Iris",
               "Lilyvalley", "Pansy", "Snowdrop", "Sunflower", "Tigerlily",
               "Tulip", "Windflower"]

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/health")
def health_check():
    return "OK", 200

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file uploaded")

        file = request.files["file"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            
            # Save original image as-is for display
            file.save(file_path)

            # Open image and apply EXIF correction and convert to RGB
            img = Image.open(file)
            img = ImageOps.exif_transpose(img)  # Fix any rotation from EXIF data

            # Preprocess image for prediction (resize and normalize)
            img_for_prediction = img.resize((224, 224))  # Resize to model input size
            img_array = np.array(img_for_prediction) / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Make prediction
            predictions = MODEL.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]  # Get class index
            confidence = float(np.max(predictions)) * 100  # Convert to percentage
            
            # Map index to class name
            species_name = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"

            return render_template("index.html", filename=filename, prediction=species_name, confidence=confidence)

        return render_template("index.html", message="Invalid file type. Please upload a PNG, JPG, JPEG, or WEBP image.")

    return render_template("index.html")

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Cloud Run will set the port
    app.run(host="0.0.0.0", port=port, debug=False)
