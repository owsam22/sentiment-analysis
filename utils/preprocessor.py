"""
Text Preprocessing Pipeline v2
- preprocess_text()     → for ML model (TF-IDF): removes stop words + stems
- raw_tokens()          → for lexicon analyzer: keeps negators like "not", "didn't"
"""

import re
import string

# Stop words that are SAFE to remove (excludes negators and intensifiers)
STOP_WORDS = {
    'a','an','the','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','for','of','in','on',
    'at','to','from','with','by','it','its','this','that','these',
    'those','i','my','me','we','our','you','your','they','their',
    'he','his','she','her','as','so','if','then','than','when',
    'while','about','above','after','before','between','into',
    'through','during','up','will','would','could','should','may',
    'might','must','can','own','same','just','because','until',
    'both','few','more','most','other','some','such','very','s',
    't','re','ll','ve','m','d','ain','also','even','still','yet',
    'only','already','always','often','sometimes','usually','here',
    'there','where','which','who','what','how','all','each','every',
    'any','few','too','much','many','well','back','way','since',
    'however','therefore','thus','hence','though','although','while',
    'and','or','but','nor','so','for','yet','either','neither',
    'am','is','are','was','were','been','being',
}

# Negators must NOT be removed — they flip sentiment
NEGATORS = {
    "not","no","never","neither","nor","nothing","nobody","nowhere",
    "dont","doesnt","didnt","wont","wouldnt","cant","cannot",
    "couldnt","shouldnt","wasnt","werent","isnt","arent","havent",
    "hasnt","hadnt","hardly","barely","scarcely","without","failed",
    "fail","fails","lack","lacking","lacks",
}


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    # Expand contractions before removing punctuation
    text = re.sub(r"didn['']t",  "didnt",    text)
    text = re.sub(r"don['']t",   "dont",     text)
    text = re.sub(r"doesn['']t", "doesnt",   text)
    text = re.sub(r"won['']t",   "wont",     text)
    text = re.sub(r"can['']t",   "cant",     text)
    text = re.sub(r"wasn['']t",  "wasnt",    text)
    text = re.sub(r"weren['']t", "werent",   text)
    text = re.sub(r"isn['']t",   "isnt",     text)
    text = re.sub(r"aren['']t",  "arent",    text)
    text = re.sub(r"haven['']t", "havent",   text)
    text = re.sub(r"hasn['']t",  "hasnt",    text)
    text = re.sub(r"hadn['']t",  "hadnt",    text)
    text = re.sub(r"wouldn['']t","wouldnt",  text)
    text = re.sub(r"couldn['']t","couldnt",  text)
    text = re.sub(r"shouldn['']t","shouldnt",text)
    text = re.sub(r"[^a-z\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> list:
    return [t for t in text.split() if len(t) > 0]


def raw_tokens(text: str) -> list:
    """
    Returns cleaned tokens INCLUDING negators.
    Used by the lexicon-based analyzer so negation context is preserved.
    """
    cleaned = clean_text(text)
    return tokenize(cleaned)


def remove_stop_words(tokens: list) -> list:
    """Remove stop words but KEEP negators."""
    return [t for t in tokens if t not in STOP_WORDS or t in NEGATORS]


def simple_stem(word: str) -> str:
    for suffix in ['ing','tion','ness','ment','ful','less','able',
                   'ible','ous','ive','est','er','ed','ly','es','s']:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[:-len(suffix)]
    return word


def stem_tokens(tokens: list) -> list:
    return [simple_stem(t) for t in tokens]


def preprocess_text(text: str, stem: bool = True) -> list:
    """
    For ML model (TF-IDF training/inference).
    Returns stemmed tokens with stop words removed.
    Negators are kept so TF-IDF bigrams like 'not_good' are possible.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stop_words(tokens)
    if stem:
        tokens = stem_tokens(tokens)
    return [t for t in tokens if len(t) > 1]


def get_preprocessing_steps(text: str) -> dict:
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    no_stop = remove_stop_words(tokens)
    stemmed = stem_tokens(no_stop)
    final = [t for t in stemmed if len(t) > 1]
    return {
        'original': text,
        'cleaned': cleaned,
        'tokens': tokens,
        'no_stop_words': no_stop,
        'stemmed': stemmed,
        'final_tokens': final,
    }
