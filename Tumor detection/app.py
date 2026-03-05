import os
import time
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# --------------------------------------------------
# Rebuild Model Architecture (MUST MATCH TRAINING)
# --------------------------------------------------

base_model = EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(380, 380, 3),
    pooling='max'
)

base_model.trainable = True  # Because you fine-tuned entire model

model = Sequential([
    base_model,
    BatchNormalization(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(4, activation="softmax")
])

# --------------------------------------------------
# Load Extracted Weights
# --------------------------------------------------

model.load_weights("model_weights.weights.h5")

print("✅ Model loaded successfully")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --------------------------------------------------
# Image Preprocessing (MUST MATCH TRAINING)
# --------------------------------------------------

def prepare_image(img_path):
    img = Image.open(img_path).convert("RGB")

    # Resize to 380x380 (IMPORTANT)
    img = img.resize((380, 380))

    img_array = np.array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # EfficientNet preprocessing
    img_array = preprocess_input(img_array)

    return img_array


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    inference_time = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            processed_image = prepare_image(filepath)

            start = time.time()
            preds = model.predict(processed_image, verbose=0)
            inference_time = f"{time.time() - start:.2f}s"

            predicted_class = class_names[np.argmax(preds)]
            confidence = round(float(np.max(preds)) * 100, 2)

            prediction = predicted_class

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           inference_time=inference_time)


if __name__ == "__main__":
    app.run(debug=True)