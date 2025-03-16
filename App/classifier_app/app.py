import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = r"static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure directory exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load TensorFlow model
MODEL_PATH = r"model/model2.h5"
MODEL = tf.keras.models.load_model(MODEL_PATH)

# Define class names (match your training dataset's class names)
class_names = ["Bluebell","Buttercup","Coltsfoot","Cowslip","Crocus",
               "Daffodil","Daisy","Dandelion","Fritillary","Iris",
               "Lilyvalley","Pansy","Snowdrop","Sunflower","Tigerlily","Tulip","Windflower"] 

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing function (matches training pipeline)
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Convert to RGB
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixels to [0, 1]
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

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
            file.save(file_path)
            
            # Preprocess image and make prediction
            img = preprocess_image(file_path)
            predictions = MODEL.predict(img)
            predicted_class = np.argmax(predictions, axis=1)[0]  # Get class index
            confidence = float(np.max(predictions)) * 100  # Convert to percentage
            
            # Map index to class name
            species_name = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"

            return render_template("index.html", filename=filename, prediction=species_name, confidence=confidence)

    return render_template("index.html")

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Cloud Run will set the port
    app.run(host="0.0.0.0", port=port, debug=False)