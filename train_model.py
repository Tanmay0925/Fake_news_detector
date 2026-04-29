"""
train_model.py — Trains the Fake News Detector
Run this once before starting the Flask app.

Dataset: WELFake / ISOT or any CSV with columns: text, label
  label: 0 = REAL, 1 = FAKE
"""

import os
import re
import string
import pickle
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── CONFIG ───────────────────────────────────────────────
DATASET_PATH = os.path.join("dataset", "news.csv")   # Change if needed
MODEL_OUT    = os.path.join("models", "fake_news_model.pkl")
VEC_OUT      = os.path.join("models", "tfidf_vectorizer.pkl")
TEST_SIZE    = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 50000
# ──────────────────────────────────────────────────────────

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


def load_dataset(path):
    print(f"[1/5] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"      Shape: {df.shape}")
    print(f"      Columns: {list(df.columns)}")

    # Flexible column mapping
    text_col = None
    label_col = None

    for col in df.columns:
        if col.lower() in ['text', 'content', 'article', 'body']:
            text_col = col
        if col.lower() in ['label', 'class', 'fake', 'target']:
            label_col = col

    if text_col is None or label_col is None:
        raise ValueError("Could not find 'text' or 'label' columns. Check your CSV.")

    df = df[[text_col, label_col]].dropna()
    df.columns = ['text', 'label']

    # Ensure label is numeric (0 = real, 1 = fake)
    if df['label'].dtype == object:
        df['label'] = df['label'].str.lower().map({'fake': 1, 'real': 0, '1': 1, '0': 0}).fillna(df['label'])
    df['label'] = df['label'].astype(int)

    print(f"      Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def train(path=DATASET_PATH):
    df = load_dataset(path)

    print("[2/5] Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)

    print("[3/5] Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['label']
    )

    print("[4/5] Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    print("[5/5] Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=RANDOM_STATE)
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"{'='*50}\n")

    # Save
    os.makedirs("models", exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    with open(VEC_OUT, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"[DONE] Model saved to  : {MODEL_OUT}")
    print(f"[DONE] Vectorizer saved: {VEC_OUT}")
    print("\nYou can now start the app with:  python app.py")


if __name__ == '__main__':
    train()
