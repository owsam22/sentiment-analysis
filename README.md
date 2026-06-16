# Customer Feedback Sentiment Analysis

ML-based system that classifies customer feedback as **positive**, **negative**, or **neutral**.
>made by [owsam22](https://samarpan-portfolio.vercel.app)
---

## Project Structure

```
sentiment_analysis/
├── app.py                  # Flask web application
├── train_model.py          # ML model training script
├── requirements.txt
├── data/
│   └── sample_reviews.csv  # Sample labeled dataset
├── models/                 # Saved trained models (generated)
├── notebooks/
│   └── sentiment_analysis.ipynb  # EDA & visualization
├── static/
│   ├── css/style.css
│   └── js/dashboard.js
├── templates/
│   └── index.html
└── utils/
    ├── preprocessor.py     # Tokenization, stop-word removal, stemming
    └── analyzer.py         # Lexicon-based sentiment scoring
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the ML model (optional — lexicon model works out of the box)
```bash
python train_model.py
```

### 3. Run the Flask dashboard
```bash
python app.py
# Open http://localhost:5000
```

### 4. Explore the Jupyter notebook
```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Predict sentiment for a single review |
| POST | `/api/batch` | Batch predict for multiple reviews |
| POST | `/api/insights` | Generate business insights from reviews |

### Example — single prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'
```

### Response
```json
{
  "sentiment": "positive",
  "confidence": 0.91,
  "confidence_pct": "91%",
  "scores": { "positive": 2, "negative": 0, "neutral": 3 },
  "token_count": 5,
  "top_tokens": ["product", "absolut", "amaz"]
}
```

---

## Preprocessing Pipeline

1. **Lowercasing** — normalize text case
2. **Punctuation & URL removal** — strip noise
3. **Tokenization** — split into word tokens
4. **Stop-word removal** — filter common words
5. **Stemming** — reduce words to root form

---

## ML Models Supported

- Logistic Regression (default, best balanced accuracy)
- Naive Bayes (fast, good for small datasets)
- Linear SVM (high accuracy on large datasets)

Switch to BERT/RoBERTa via `transformers` for state-of-the-art results.

---

## Extending with Your Own Data

Replace `data/sample_reviews.csv` with your dataset. Required columns:
- `text` — the review text
- `sentiment` — one of `positive`, `negative`, `neutral`

Then re-run `python train_model.py` to retrain.

---

## Tech Stack

- **Backend**: Python, Flask
- **ML**: scikit-learn, NLTK
- **Visualization**: Chart.js, Matplotlib, Seaborn, WordCloud
- **Frontend**: HTML5, CSS3, Vanilla JS
