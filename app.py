from flask import Flask, request, jsonify, render_template
from utils.preprocessor import preprocess_text
from utils.analyzer import SentimentAnalyzer

app = Flask(__name__)
analyzer = SentimentAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    tokens = preprocess_text(text)
    result = analyzer.predict(text, tokens)
    return jsonify(result)

@app.route('/api/batch', methods=['POST'])
def batch_predict():
    data = request.get_json()
    reviews = data.get('reviews', [])
    results = []
    for review in reviews:
        tokens = preprocess_text(review)
        result = analyzer.predict(review, tokens)
        result['original_text'] = review
        results.append(result)
    return jsonify({'results': results, 'total': len(results)})

@app.route('/api/insights', methods=['POST'])
def insights():
    data = request.get_json()
    reviews = data.get('reviews', [])
    results = []
    for review in reviews:
        tokens = preprocess_text(review)
        result = analyzer.predict(review, tokens)
        results.append(result)
    insights = analyzer.generate_insights(results)
    return jsonify(insights)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
