"""
Sentiment Analyzer
Supports: lexicon-based analysis + scikit-learn ML model
"""

from collections import Counter

POSITIVE_WORDS = {
    'love','great','excellent','amazing','fantastic','good','best','happy',
    'wonderful','outstanding','exceptional','superb','awesome','perfect','nice',
    'smooth','fast','quick','helpful','easy','recommend','pleased','satisfied',
    'brilliant','delight','impress','enjoy','benefit','clean','clear','strong',
    'beautiful','elegant','reliable','efficient','effective','professional',
    'friendly','polite','responsive','innovative','creative','intuitive',
    'affordable','valuable','premium','quality','superior','flawless','seamless',
    'convenient','comfortable','stylish','durable','robust','versatile','fun',
    'exciting','refreshing','charming','pleasant','sparkling','vivid','bright'
}

NEGATIVE_WORDS = {
    'terrible','bad','awful','horrible','disappoint','poor','slow','broke',
    'broken','crash','damage','unclear','unhelpful','frustrat','delay','wait',
    'unacceptable','worst','problem','issue','fail','defect','error','bug',
    'expensive','overpriced','waste','useless','mislead','cheat','fake','lie',
    'rude','ignorant','dismiss','neglect','abandon','ignore','confuse','difficult',
    'complicated','clunky','outdated','unstable','unreliable','inconsistent',
    'disappear','missing','lost','wrong','incorrect','inaccurate','incomplete',
    'dirty','messy','noisy','loud','uncomfortable','ugly','cheap','flimsy',
    'fragile','weak','narrow','limited','restrictive','annoying','aggravat'
}

INTENSIFIERS = {'very','really','extremely','absolutely','totally','incredibly','so'}
NEGATORS    = {'not','never','no','nobody','nothing','neither','nor','cannot','cant','dont','wont','isnt','wasnt'}


class SentimentAnalyzer:
    """
    Lexicon-based sentiment analyzer with confidence scoring.
    Replace predict() internals with a trained sklearn/transformers model
    for production use.
    """

    def __init__(self):
        self.pos_words = POSITIVE_WORDS
        self.neg_words = NEGATIVE_WORDS

    def _score_tokens(self, tokens: list) -> dict:
        pos, neg, neu = 0, 0, 0
        negate = False
        intensify = False

        for token in tokens:
            if token in NEGATORS:
                negate = True
                continue
            if token in INTENSIFIERS:
                intensify = True
                continue

            boost = 2 if intensify else 1
            matched_pos = token in self.pos_words or any(
                token.startswith(w[:5]) for w in self.pos_words if len(w) >= 5)
            matched_neg = token in self.neg_words or any(
                token.startswith(w[:5]) for w in self.neg_words if len(w) >= 5)

            if matched_pos:
                if negate:
                    neg += boost
                else:
                    pos += boost
                negate = False
                intensify = False
            elif matched_neg:
                if negate:
                    pos += boost
                else:
                    neg += boost
                negate = False
                intensify = False
            else:
                neu += 1

        return {'positive': pos, 'negative': neg, 'neutral': neu}

    def _classify(self, scores: dict, token_count: int) -> tuple:
        pos, neg = scores['positive'], scores['negative']

        if pos == 0 and neg == 0:
            return 'neutral', 0.60

        total_signal = pos + neg
        if total_signal == 0:
            return 'neutral', 0.60

        if pos > neg * 1.5:
            conf = min(0.98, 0.65 + (pos / max(token_count, 1)) * 0.50)
            return 'positive', round(conf, 2)
        elif neg > pos * 1.5:
            conf = min(0.98, 0.65 + (neg / max(token_count, 1)) * 0.50)
            return 'negative', round(conf, 2)
        elif pos > neg:
            return 'positive', 0.62
        elif neg > pos:
            return 'negative', 0.62
        else:
            return 'neutral', 0.58

    def predict(self, original_text: str, tokens: list) -> dict:
        scores = self._score_tokens(tokens)
        sentiment, confidence = self._classify(scores, len(tokens))

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'confidence_pct': f"{int(confidence * 100)}%",
            'scores': scores,
            'token_count': len(tokens),
            'top_tokens': tokens[:10],
        }

    def generate_insights(self, results: list) -> dict:
        total = len(results)
        if total == 0:
            return {'error': 'No results to analyze'}

        counts = Counter(r['sentiment'] for r in results)
        pos_pct = round(counts['positive'] / total * 100, 1)
        neg_pct = round(counts['negative'] / total * 100, 1)
        neu_pct = round(counts['neutral'] / total * 100, 1)
        avg_conf = round(sum(r['confidence'] for r in results) / total * 100, 1)
        nsi = counts['positive'] - counts['negative']

        insights = []
        if pos_pct >= 60:
            insights.append('Strong positive sentiment — customers are generally satisfied.')
        elif pos_pct < 40:
            insights.append('High negative sentiment — immediate action required.')
        if neg_pct >= 30:
            insights.append('Significant negative feedback detected — investigate root causes.')
        if neu_pct >= 35:
            insights.append('Many neutral reviews — opportunity to convert fence-sitters.')

        return {
            'total': total,
            'positive_count': counts['positive'],
            'negative_count': counts['negative'],
            'neutral_count': counts['neutral'],
            'positive_pct': pos_pct,
            'negative_pct': neg_pct,
            'neutral_pct': neu_pct,
            'avg_confidence': avg_conf,
            'net_sentiment_index': nsi,
            'csat_score': pos_pct,
            'insights': insights,
        }
