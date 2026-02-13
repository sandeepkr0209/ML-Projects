# 🎵 Spotify Music Recommendation System (Content-Based)

A machine learning project that recommends similar songs based on audio features using a **content-based filtering** approach. The system uses Spotify track features and a **Nearest Neighbors model with cosine similarity** to suggest songs that sound similar to a given track.

---

## 📌 Project Overview

This project builds a music recommender using Spotify track data. Instead of relying on user behavior, it compares songs using their intrinsic audio features such as:

* danceability
* energy
* loudness
* speechiness
* acousticness
* instrumentalness
* liveness
* valence
* tempo

A Nearest Neighbors model is trained on normalized features to find the most similar tracks.

---

## 🗂 Dataset

* File used: `dataset.csv`
* Size: ~114,000 tracks
* Features include:

  * Track metadata (artist, album, name, genre)
  * Popularity & duration
  * Audio features from Spotify

---

## ⚙️ Workflow

### 1️⃣ Data Loading

```python
df = pd.read_csv('dataset.csv')
```

### 2️⃣ Data Cleaning

* Dropped unnecessary columns:

  * `Unnamed: 0`
  * `track_id`
* Removed rows with missing values (only 1 row affected)
* Removed duplicate records (~577 rows)

### 3️⃣ Feature Selection

Selected only audio features for similarity:

```python
features = [
'danceability','energy','loudness','speechiness',
'acousticness','instrumentalness','liveness',
'valence','tempo'
]
```

---

### 4️⃣ Feature Scaling

Standardized features using StandardScaler:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### 5️⃣ Model Training

Used Nearest Neighbors with cosine similarity:

```python
from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors(
    metric='cosine',
    algorithm='brute',
    n_neighbors=10
)

model.fit(X_scaled)
```

---

## 🎯 Recommendation Function

```python
def recommend(song_name, n=5):

    idx = df[df['track_name'] == song_name].index[0]

    distances, indices = model.kneighbors(
        [X_scaled[idx]],
        n_neighbors=n+1
    )

    rec_songs = df.iloc[indices[0][1:]][
        ['track_name','artists','track_genre']
    ]

    return rec_songs
```

### ✅ Example

```python
recommend("Believer")
```

Returns top similar songs based on audio features.

---

## 💾 Model Saving

Model and scaler are saved using joblib:

```python
import joblib

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

---

## 🧰 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook

---

## 🚀 How to Run

1. Clone the repository
2. Place `dataset.csv` in the project folder
3. Install dependencies:

```bash
pip install pandas numpy scikit-learn joblib
```

4. Run the notebook or script
5. Call:

```python
recommend("song_name")
```

---

## 📈 Possible Improvements

* Add artist + genre weighting
* Use PCA for dimensionality reduction
* Build a web app using Flask/Streamlit
* Combine with collaborative filtering
* Add fuzzy matching for song names
* Handle multiple songs with same title

---

## 📄 License

This project is for educational and portfolio purposes.

---

## 🙌 Author

Sandeep Kumar — AI/ML Project Portfolio
