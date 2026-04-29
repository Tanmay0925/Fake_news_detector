from flask import Flask, render_template, request, jsonify
import pickle
import os
import re
import string
import nltk

# Download required NLTK data (only first time)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = os.path.join("models", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

model = None
vectorizer = None

def load_model():
    global model, vectorizer
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        print("[INFO] Model and vectorizer loaded successfully.")
    except FileNotFoundError:
        print("[WARNING] Model files not found. Please run train_model.py first.")

def clean_text(text):
    """Clean and preprocess the input text."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({
            "error": "Model not loaded. Please run train_model.py first.",
            "status": "error"
        }), 500

    data = request.get_json()
    news_text = data.get('text', '').strip()

    if not news_text:
        return jsonify({"error": "No text provided.", "status": "error"}), 400

    if len(news_text.split()) < 10:
        return jsonify({"error": "Please provide at least 10 words for accurate detection.", "status": "error"}), 400

    cleaned = clean_text(news_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]

    confidence = round(float(max(proba)) * 100, 2)
    label = "FAKE" if prediction == 1 else "REAL"

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "status": "success",
        "word_count": len(news_text.split())
    })

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    load_model()
    app.run(debug=True)
