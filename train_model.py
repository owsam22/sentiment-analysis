"""
ML Model Training Script
Trains Logistic Regression, Naive Bayes, and SVM classifiers
on labeled customer feedback data.

Usage:
    python train_model.py

Requirements:
    pip install scikit-learn pandas numpy joblib
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils.preprocessor import preprocess_text

# ─── Sample dataset (replace with your real CSV) ─────────────────────────────
SAMPLE_DATA = [
    ("Absolutely love this product! Best purchase this year.", "positive"),
    ("Excellent customer support! Resolved my issue in hours.", "positive"),
    ("Amazing quality, smooth delivery, very happy!", "positive"),
    ("Great value for money, highly recommend.", "positive"),
    ("Outstanding build quality. Exceeded expectations.", "positive"),
    ("Smooth checkout and fast shipping. Very pleased!", "positive"),
    ("Couldn't be happier. Will definitely buy again.", "positive"),
    ("The subscription plan offers great flexibility.", "positive"),
    ("Returns process was easy and hassle-free.", "positive"),
    ("Brilliant product, works perfectly right out of the box.", "positive"),

    ("Delivery took forever and the package was damaged.", "negative"),
    ("Terrible quality, broke after two days.", "negative"),
    ("The app crashes frequently. Very frustrating.", "negative"),
    ("Waited 3 weeks with no shipping updates. Unacceptable.", "negative"),
    ("Support team was unhelpful and dismissive.", "negative"),
    ("Product looked completely different from the photos.", "negative"),
    ("Instructions were unclear, assembly took hours.", "negative"),
    ("Very disappointed. Would not recommend to anyone.", "negative"),
    ("Worst purchase I have ever made. Total waste of money.", "negative"),
    ("Item arrived broken with no protective packaging.", "negative"),

    ("The item works as described. Nothing special.", "neutral"),
    ("Average experience. Not bad but could be improved.", "neutral"),
    ("Decent product. Does what it says on the tin.", "neutral"),
    ("Normal delivery, normal experience.", "neutral"),
    ("The product is okay. Nothing extraordinary.", "neutral"),
    ("It is fine for the price I suppose.", "neutral"),
    ("Works as expected. No complaints, no praise.", "neutral"),
    ("Standard quality, standard packaging, standard delivery.", "neutral"),
    ("Does the job. Not the best, not the worst.", "neutral"),
    ("Met my basic expectations. Nothing more, nothing less.", "neutral"),
]


def load_data(csv_path: str = None):
    """Load data from CSV or use built-in sample data."""
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
        return df['text'].tolist(), df['sentiment'].tolist()
    else:
        print("Using built-in sample data (30 examples). Provide a CSV for better accuracy.")
        texts, labels = zip(*SAMPLE_DATA)
        return list(texts), list(labels)


def preprocess_corpus(texts: list) -> list:
    """Preprocess all texts and join tokens back into strings for TF-IDF."""
    return [' '.join(preprocess_text(t)) for t in texts]


def build_pipelines() -> dict:
    """Return a dict of named sklearn pipelines."""
    return {
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]),
        'Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('clf', MultinomialNB(alpha=0.5)),
        ]),
        'Linear SVM': Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ('clf', LinearSVC(C=1.0, max_iter=2000, random_state=42)),
        ]),
    }


def evaluate_models(pipelines: dict, X_train, X_test, y_train, y_test) -> str:
    """Train and evaluate all pipelines, return name of best model."""
    best_name, best_acc = None, 0
    results = {}

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"\n{'='*50}")
        print(f"  {name}  —  Accuracy: {acc:.2%}")
        print('='*50)
        print(classification_report(y_test, y_pred, target_names=['negative','neutral','positive']))

        if acc > best_acc:
            best_acc = acc
            best_name = name

    print(f"\n✅ Best model: {best_name} ({best_acc:.2%})")
    return best_name


def save_model(pipeline, name: str, path: str = 'models/sentiment_model.pkl'):
    """Persist the trained pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Model saved → {path}")

    meta = {'model_name': name, 'classes': ['negative', 'neutral', 'positive']}
    with open(path.replace('.pkl', '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def main():
    print("Customer Feedback Sentiment Analysis — Model Training\n")

    texts, labels = load_data()
    processed = preprocess_corpus(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        processed, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    pipelines = build_pipelines()
    best_name = evaluate_models(pipelines, X_train, X_test, y_train, y_test)

    best_pipe = pipelines[best_name]
    save_model(best_pipe, best_name)

    # Quick demo prediction
    print("\n── Demo Predictions ──")
    samples = [
        "This is the best product I have ever bought!",
        "Terrible experience, never buying again.",
        "It is okay, nothing special.",
    ]
    for s in samples:
        proc = ' '.join(preprocess_text(s))
        pred = best_pipe.predict([proc])[0]
        print(f"  '{s[:50]}...' → {pred.upper()}")


if __name__ == '__main__':
    main()
