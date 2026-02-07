# 📧 Email Spam Detector — ML + Flask Deployment

A production-ready Email Spam Detection web application built using **Machine Learning (TF-IDF + Logistic Regression)** and deployed with **Flask**.
The system classifies email text as **Spam** or **Not Spam** with high accuracy and has been validated against real Gmail inbox data.

---

## 🚀 Project Overview

This project implements an end-to-end spam email classification system:

* Text preprocessing and normalization
* Feature extraction using TF-IDF with n-grams
* Supervised ML classification using Logistic Regression
* Model packaging with Scikit-Learn Pipeline
* Web interface built with Flask
* Real-time prediction with probability score
* Threshold-based spam decision to reduce false positives

The trained model was also tested on real inbox emails and matched Gmail’s spam classification with high consistency.

---

## 🧠 Model Architecture

Pipeline:

Text Input
→ Cleaning (regex normalization)
→ TF-IDF Vectorization (1–2 grams, 5000 features)
→ Logistic Regression (balanced class weights)
→ Spam Probability Output

---

## 🔬 Preprocessing Steps

Training data preprocessing included:

* Lowercasing text
* Removing numbers
* Removing punctuation
* Removing extra whitespace
* Stopword removal
* Lemmatization
* Tokenization (NLTK)

Deployment preprocessing mirrors the same normalization logic to ensure prediction consistency.

---

## ⚙️ Model Configuration

```
Vectorizer: TfidfVectorizer
max_features = 5000
ngram_range = (1,2)
stop_words = english

Classifier: LogisticRegression
max_iter = 1000
class_weight = balanced
```

Saved as:

```
spam_classifier_pipeline.pkl
label_encoder.pkl
```

---

## 📊 Evaluation Metrics (Test Set)

Accuracy: **0.9808**

| Class        | Precision | Recall | F1   |
| ------------ | --------- | ------ | ---- |
| Not Spam (0) | 0.98      | 0.97   | 0.98 |
| Spam (1)     | 0.98      | 0.99   | 0.98 |

Macro Avg F1: **0.98**
Weighted Avg F1: **0.98**

Test samples: **16,690**

---

## 🧪 Real-World Validation

The trained model was tested on real personal email data:

* ✅ Correctly classified all Gmail-flagged spam emails
* ✅ Correctly classified normal inbox emails
* ✅ No observed false negatives in manual testing set
* ✅ Strong real-world generalization performance

(This indicates good dataset coverage and strong feature separation.)

---

## 📉 Confusion Matrix

*Add your values here if available:*

```
[[ TN , FP ],
 [ FN , TP ]]
```

Generate with:

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
```

---

## 🌐 Web App Features

* Paste email text for instant prediction
* Spam probability score shown
* Adjustable spam threshold
* Clean UI interface
* Flask backend inference
* Pipeline model loading via joblib

---

## 🖥️ Project Structure

```
spam-detector/
│
├── app.py
├── requirements.txt
├── spam_classifier_pipeline.pkl
├── label_encoder.pkl
│
├── templates/
│   └── index.html
│
└── training_notebook.ipynb
```

---

## ▶️ Run Locally

Install dependencies:

```
pip install -r requirements.txt
```

Run server:

```
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

## 📦 Requirements

```
flask
scikit-learn
pandas
numpy
scipy
joblib
```

(Use same sklearn version as training environment for compatibility.)

---

## 🎯 Decision Logic

Prediction uses probability threshold instead of raw class:

```
if spam_probability >= 0.70 → SPAM
else → NOT SPAM
```

This reduces false positives in borderline promotional emails.

---

## 🔮 Future Improvements

* Add explanation of top spam keywords
* Add SHAP feature importance
* Batch email classification
* Email file upload (.txt / .eml)
* API endpoint for external apps
* Online learning feedback loop
* Docker containerization
* Cloud deployment pipeline

---

## 🏷️ Suggested Resume Description

Built a machine learning spam email classifier using TF-IDF and Logistic Regression achieving 98% accuracy, deployed as a Flask web application with real-time prediction and probability scoring, validated against real inbox data.

---

## 📄 License

Educational / Project Use
