from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# ---------- LOAD ----------
pipeline = joblib.load("spam_classifier_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# ---------- SAME CLEANING AS TRAINING ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    raw_text = request.form["email_text"]

    processed = clean_text(raw_text)

    # ---- probability first ----
    prob = pipeline.predict_proba([processed])[0]
    spam_prob = prob[1]

    # ---- custom threshold ----
    THRESHOLD = 0.70

    if spam_prob >= THRESHOLD:
        label = "SPAM 🚨"
    else:
        label = "NOT SPAM ✅"

    return render_template(
        "index.html",
        prediction=label,
        probability=f"{spam_prob:.2f}",
        threshold=THRESHOLD,
        text=raw_text
    )


if __name__ == "__main__":
    app.run()
