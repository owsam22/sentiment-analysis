"""
Text Preprocessing Pipeline
Handles: tokenization, stop-word removal, stemming/lemmatization
"""

import re
import string

# Basic English stop words (install nltk for a full list)
STOP_WORDS = {
    'a','an','the','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','but','and','or','for',
    'of','in','on','at','to','from','with','by','it','its',
    'this','that','these','those','i','my','me','we','our',
    'you','your','they','their','he','his','she','her','as',
    'so','if','then','than','when','while','about','above',
    'after','before','between','into','through','during','up',
    'will','would','could','should','may','might','must','can',
    'not','no','nor','only','own','same','just','because','as',
    'until','both','few','more','most','other','some','such',
    'very','s','t','don','won','re','ll','ve','m','d','ain'
}

def clean_text(text: str) -> str:
    """Lowercase and remove special characters."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)             # remove mentions/hashtags
    text = re.sub(r'[^a-z\s]', ' ', text)             # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> list:
    """Split text into word tokens."""
    return text.split()

def remove_stop_words(tokens: list) -> list:
    """Filter out common stop words."""
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

def simple_stem(word: str) -> str:
    """Rule-based stemmer (use nltk PorterStemmer for production)."""
    suffixes = ['ing', 'tion', 'ness', 'ment', 'ful', 'less', 'able',
                'ible', 'ous', 'ive', 'est', 'er', 'ed', 'ly', 'es', 's']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[:-len(suffix)]
    return word

def stem_tokens(tokens: list) -> list:
    """Apply stemming to each token."""
    return [simple_stem(t) for t in tokens]

def preprocess_text(text: str, stem: bool = True) -> list:
    """
    Full preprocessing pipeline.
    Returns list of cleaned tokens.

    Steps:
    1. Clean / normalize text
    2. Tokenize
    3. Remove stop words
    4. Stem tokens (optional)
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stop_words(tokens)
    if stem:
        tokens = stem_tokens(tokens)
    return [t for t in tokens if len(t) > 2]

def get_preprocessing_steps(text: str) -> dict:
    """Return step-by-step preprocessing output for display."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    no_stop = remove_stop_words(tokens)
    stemmed = stem_tokens(no_stop)
    final = [t for t in stemmed if len(t) > 2]

    return {
        'original': text,
        'cleaned': cleaned,
        'tokens': tokens,
        'no_stop_words': no_stop,
        'stemmed': stemmed,
        'final_tokens': final
    }
