"""
Sentiment Analyzer — v2
Fixes:
  - Proper negation handling ("not good", "didn't like", "wasn't happy")
  - Phrase-level patterns before token splitting
  - Stronger lexicon with "liked", "expected", "good", etc.
  - Mixed-sentiment detection for nuanced reviews
  - "didn't like", "not as expected" → negative
"""

import re
from collections import Counter

# ── Phrase-level patterns (checked BEFORE tokenization) ──────────────────────
# Format: (regex_pattern, sentiment_delta)  delta: +N = pos, -N = neg
PHRASE_PATTERNS = [
    # Strong negations of positive phrases
    (r"\bnot (good|great|nice|happy|satisfied|pleased|worth|recommend)\b", -2),
    (r"\bnot as (good|great|expected|described|advertised|shown|pictured)\b", -2),
    (r"\bdidn['']?t (like|enjoy|work|fit|last|help|expect|receive)\b", -2),
    (r"\bdoesn['']?t (work|fit|help|last|make sense)\b", -2),
    (r"\bwasn['']?t (happy|good|great|satisfied|impressed|worth it)\b", -2),
    (r"\bwon['']?t (recommend|buy|order|use)\b", -2),
    (r"\bcan['']?t (recommend|use|believe|trust)\b", -2),
    (r"\bno (good|use|value|quality|help|support)\b", -2),
    (r"\bnever (again|buy|recommend|order|use)\b", -2),
    (r"\bfar from (good|great|perfect|expected|ideal)\b", -2),
    (r"\ba (waste|disappointment|letdown|joke|disaster)\b", -2),
    (r"\bcomplete (waste|rubbish|disappointment|failure|junk)\b", -3),
    (r"\bnot worth\b", -2),
    (r"\bwaste of (money|time)\b", -3),
    (r"\bfell apart\b", -2),
    (r"\bfell short\b", -2),
    (r"\bbelow (expectations?|average|par|standard)\b", -2),
    (r"\bless than (expected|advertised|described)\b", -2),
    # Negations of negative phrases (double negative = mild positive)
    (r"\bnot (bad|terrible|awful|horrible|disappointing)\b", +1),
    (r"\bno (complaints?|issues?|problems?)\b", +1),
    # Strong positive phrases
    (r"\bexceeded (my )?expectations?\b", +3),
    (r"\bhighly recommend\b", +3),
    (r"\bbest (purchase|buy|product|decision)\b", +3),
    (r"\babsolutely (love|perfect|amazing|wonderful)\b", +3),
    (r"\b(very |really |extremely )?(happy|satisfied|pleased) with\b", +2),
    (r"\bworks (perfectly|great|like a charm|flawlessly)\b", +2),
    (r"\bfast (shipping|delivery|response)\b", +1),
    (r"\bgreat (value|quality|service|product|experience)\b", +2),
    (r"\bgood (value|quality|service|product|experience)\b", +1),
    (r"\bwill (definitely )?buy again\b", +2),
    (r"\bworth (every penny|the money|the price)\b", +2),
    (r"\bexactly as (described|advertised|expected|shown)\b", +2),
    (r"\bas expected\b", +1),
    # Strong negative phrases
    (r"\bvery (disappointed|frustrating|poor|bad)\b", -3),
    (r"\bextremely (disappointed|frustrated|poor|bad|slow)\b", -3),
    (r"\bterrible (quality|service|experience|product)\b", -3),
    (r"\bpoor (quality|service|experience|product|packaging)\b", -2),
    (r"\bbroke (after|within|in)\b", -3),
    (r"\bstopped working\b", -3),
    (r"\bdamaged (on arrival|in transit|during shipping)\b", -3),
    (r"\blooks (nothing like|completely different|fake)\b", -2),
    (r"\btook (forever|too long|weeks|months)\b", -2),
    (r"\bno (updates?|communication|response|tracking)\b", -2),
]

# ── Token-level lexicons ──────────────────────────────────────────────────────
POSITIVE_TOKENS = {
    'love','great','excellent','amazing','fantastic','wonderful','outstanding',
    'exceptional','superb','awesome','perfect','brilliant','delighted','pleased',
    'satisfied','happy','enjoy','enjoyed','liked','like','recommend','impressive',
    'flawless','seamless','reliable','durable','sturdy','elegant','beautiful',
    'comfortable','convenient','efficient','effective','responsive','helpful',
    'friendly','polite','fast','quick','smooth','easy','affordable','valuable',
    'premium','superior','stylish','fun','exciting','refreshing','charming',
    'pleasant','sparkling','vivid','bright','worth','accurate','genuine','authentic'
}

NEGATIVE_TOKENS = {
    'terrible','bad','awful','horrible','poor','disappointed','disappointing',
    'frustrating','frustration','broken','broke','damaged','defective','faulty',
    'useless','worthless','waste','junk','rubbish','cheap','flimsy','fragile',
    'slow','delayed','late','missing','lost','wrong','incorrect','inaccurate',
    'misleading','fake','counterfeit','rude','unhelpful','dismissive','ignored',
    'crash','crashes','crashing','error','bug','unstable','unreliable','difficult',
    'confusing','complicated','ugly','dirty','noisy','uncomfortable','overpriced',
    'expensive','scam','fraud','lie','lied','cheat','cheated','regret','regretted',
    'annoying','irritating','aggressive','negligent','unacceptable','unresponsive',
    'incomplete','unclear','vague','mislead','misled','deteriorate','deteriorated'
}

NEGATORS = {
    "not","no","never","neither","nor","nothing","nobody","nowhere",
    "don't","doesn't","didn't","won't","wouldn't","can't","cannot",
    "couldn't","shouldn't","wasn't","weren't","isn't","aren't","haven't",
    "hasn't","hadn't","hardly","barely","scarcely","lack","lacking","lacks",
    "without","minus","failed","fail","fails"
}

INTENSIFIERS = {
    "very":1.5, "really":1.5, "extremely":2.0, "absolutely":2.0,
    "totally":1.5, "incredibly":2.0, "so":1.3, "super":1.5,
    "highly":1.5, "deeply":1.5, "completely":1.5, "utterly":2.0,
    "truly":1.5, "quite":1.2, "pretty":1.2, "somewhat":0.7,
    "slightly":0.6, "a bit":0.7, "a little":0.6, "kind of":0.7,
    "sort of":0.7
}

DOWNGRADERS = {"somewhat", "slightly", "a bit", "a little", "kind of", "sort of", "quite", "fairly", "rather"}


def score_phrases(text: str) -> float:
    """Score text using phrase-level regex patterns. Returns float delta."""
    text_lower = text.lower()
    delta = 0.0
    for pattern, weight in PHRASE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        delta += weight * len(matches)
    return delta


def score_tokens(tokens: list) -> dict:
    """Token-level scoring with negation window and intensifiers."""
    pos, neg = 0.0, 0.0
    n = len(tokens)
    negation_window = 0   # counts down — within this many tokens, flip polarity
    intensity = 1.0

    for i, token in enumerate(tokens):
        # Detect negators
        if token in NEGATORS:
            negation_window = 3  # next 3 tokens are negated
            continue

        # Detect intensifiers
        if token in INTENSIFIERS:
            intensity = INTENSIFIERS[token]
            continue

        # Detect downgraders
        if token in DOWNGRADERS:
            intensity = 0.65
            continue

        is_negated = negation_window > 0
        negation_window = max(0, negation_window - 1)

        matched_pos = token in POSITIVE_TOKENS
        matched_neg = token in NEGATIVE_TOKENS

        if matched_pos:
            if is_negated:
                neg += 1.5 * intensity
            else:
                pos += 1.0 * intensity
            intensity = 1.0

        elif matched_neg:
            if is_negated:
                pos += 0.5 * intensity   # double-negative = weak positive
            else:
                neg += 1.0 * intensity
            intensity = 1.0
        else:
            intensity = 1.0  # reset if no match

    return {'positive': round(pos, 2), 'negative': round(neg, 2)}


class SentimentAnalyzer:

    def predict(self, original_text: str, tokens: list) -> dict:
        phrase_delta = score_phrases(original_text)
        token_scores = score_tokens(tokens)

        # Combine phrase-level and token-level scores
        combined_pos = token_scores['positive'] + max(0, phrase_delta)
        combined_neg = token_scores['negative'] + max(0, -phrase_delta)

        sentiment, confidence = self._classify(combined_pos, combined_neg, len(tokens))

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'confidence_pct': f"{int(confidence * 100)}%",
            'scores': {
                'positive': round(combined_pos, 2),
                'negative': round(combined_neg, 2),
                'phrase_delta': round(phrase_delta, 2),
            },
            'token_count': len(tokens),
            'top_tokens': tokens[:10],
        }

    def _classify(self, pos: float, neg: float, token_count: int) -> tuple:
        total = pos + neg
        if total < 0.5:
            return 'neutral', 0.60

        ratio = pos / total if total > 0 else 0.5

        if ratio >= 0.70:
            conf = min(0.97, 0.60 + ratio * 0.40 + min(total, 5) * 0.02)
            return 'positive', round(conf, 2)
        elif ratio <= 0.30:
            conf = min(0.97, 0.60 + (1 - ratio) * 0.40 + min(total, 5) * 0.02)
            return 'negative', round(conf, 2)
        else:
            # Mixed — lean toward dominant side with lower confidence
            if pos > neg:
                return 'positive', round(0.55 + ratio * 0.10, 2)
            elif neg > pos:
                return 'negative', round(0.55 + (1 - ratio) * 0.10, 2)
            else:
                return 'neutral', 0.58

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
            insights.append('Significant negative feedback — investigate root causes.')
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


# Monkey-patch score_phrases at bottom to add missing patterns handled in v2
_EXTRA_PATTERNS = [
    (r"\bdidn['']?t love\b", -1),
    (r"\b(liked|like) (it|some) but\b", -1),   # "liked it but..." = mixed
    (r"\bnot what (was|is|were|are) (shown|described|advertised|expected|pictured)\b", -2),
    (r"\b(average|mediocre|ordinary|so-so|bland)\b", -1),
]

_orig_score_phrases = score_phrases
def score_phrases(text: str) -> float:
    delta = _orig_score_phrases(text)
    text_lower = text.lower()
    for pattern, weight in _EXTRA_PATTERNS:
        import re as _re
        matches = _re.findall(pattern, text_lower)
        delta += weight * len(matches)
    return delta
