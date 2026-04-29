"""
utils/text_cleaner.py
Reusable text preprocessing functions.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

_stop_words = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove punctuation
    5. Remove digits
    6. Tokenize, remove stopwords, lemmatize
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [_lemmatizer.lemmatize(w) for w in tokens if w not in _stop_words and len(w) > 2]
    return " ".join(tokens)


def word_count(text: str) -> int:
    return len(text.strip().split())
