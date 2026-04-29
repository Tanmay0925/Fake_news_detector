# 🔍 Fake News Detection using NLP

A web-based application that detects whether a news article is **REAL or FAKE** using **Natural Language Processing (NLP)** and **Machine Learning**. Built with Python, Flask, scikit-learn, and NLTK.

> **3rd Year Major Project — B.Tech CSE**

---

## 📌 Project Overview

The spread of misinformation online is a growing problem. This project tackles it by building a machine learning model that classifies news articles using text analysis techniques. The user can paste any news headline or article into the web interface and get an instant prediction with a confidence score.

---

## ✨ Features

- Clean, interactive web interface built with HTML/CSS/JS
- REST API endpoint (`/predict`) for classification
- Full NLP pipeline: cleaning → stopword removal → lemmatization → TF-IDF
- Logistic Regression classifier with probability scores
- Works with any news dataset in CSV format
- Jupyter notebook included for EDA and model exploration

---

## 🛠️ Tech Stack

| Layer       | Technology                      |
|-------------|----------------------------------|
| Backend     | Python, Flask                   |
| ML Model    | scikit-learn (Logistic Regression) |
| NLP         | NLTK (stopwords, lemmatizer)    |
| Vectorizer  | TF-IDF (bigrams, 50k features)  |
| Frontend    | HTML5, CSS3, Vanilla JS         |
| Data        | Pandas, NumPy                   |

---

## 📁 Project Structure

```
fake-news-detection/
│
├── app.py                    # Flask application (main entry point)
├── train_model.py            # Script to train and save the model
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── templates/
│   ├── index.html            # Home page with prediction UI
│   └── about.html            # About / project info page
│
├── utils/
│   ├── __init__.py
│   └── text_cleaner.py       # Reusable NLP preprocessing functions
│
├── models/                   # Saved model files (generated after training)
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── dataset/
│   ├── README.md             # Instructions on where to get the dataset
│   └── news.csv              # (You add this — see Dataset section below)
│
└── notebooks/
    └── EDA_and_Training.ipynb  # Exploratory Data Analysis notebook
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
cd fake-news-detection
```

### 2. Create and Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Download the **WELFake dataset** from Kaggle:
- https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

Rename/move the file to: `dataset/news.csv`

The CSV must have columns named `text` and `label` (0 = Real, 1 = Fake).
See `dataset/README.md` for more options including the ISOT dataset.

### 5. Train the Model

```bash
python train_model.py
```

This will print accuracy, classification report, and save:
- `models/fake_news_model.pkl`
- `models/tfidf_vectorizer.pkl`

### 6. Run the Web Application

```bash
python app.py
```

Open your browser and visit: **http://127.0.0.1:5000**

---

## 🧠 How It Works

```
Input Text
    ↓
Text Cleaning (lowercase, remove URLs, punctuation, digits)
    ↓
Stopword Removal (NLTK English stopwords)
    ↓
Lemmatization (WordNetLemmatizer)
    ↓
TF-IDF Vectorization (50,000 features, unigrams + bigrams)
    ↓
Logistic Regression Classifier
    ↓
Output: REAL / FAKE + Confidence Score
```

---

## 📊 Model Performance

> Tested on WELFake dataset (~72,000 articles)

| Metric    | Value   |
|-----------|---------|
| Accuracy  | ~96–98% |
| Precision | ~97%    |
| Recall    | ~96%    |
| F1 Score  | ~96%    |

*Results may vary depending on dataset used.*

---

## 📡 API Usage

You can also call the prediction endpoint directly:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Paste your news article text here"}'
```

**Response:**
```json
{
  "prediction": "FAKE",
  "confidence": 94.21,
  "status": "success",
  "word_count": 48
}
```

---

## 📓 Jupyter Notebook

For full data exploration and visualization:

```bash
pip install jupyter wordcloud matplotlib seaborn
jupyter notebook notebooks/EDA_and_Training.ipynb
```

---

## 🚀 Future Improvements

- [ ] Add BERT / transformer-based model for higher accuracy
- [ ] Support URL-based article scraping
- [ ] Add multilingual news support
- [ ] Deploy on Heroku / Render
- [ ] Add user feedback mechanism to improve model

---

## 👨‍💻 Contributors

| Name | Roll Number |
|------|-------------|
| [Your Name] | [Your Roll No.] |
| [Team Member 2] | [Roll No.] |
| [Team Member 3] | [Roll No.] |

**Guided by:** [Supervisor Name], [Department Name]

---

## 📄 License

This project is for educational purposes as part of a university major project.

---

*Made with ❤️ for B.Tech CSE Major Project*
