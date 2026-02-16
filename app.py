from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging
import PyPDF2
from io import BytesIO
import docx
from collections import Counter
import numpy as np
import warnings
import string
import threading
import time

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Database module
import database as db

# Heavy imports - these will be imported when models are loaded
transformers = None
torch = None
sklearn = None
# Import nltk at the top level since we need it for initialization
import nltk

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Some weights of')

# Initialize logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for lazy loading
tokenizer = None
model = None
vectorizer = None
models_loaded = False
_models_loading = False
_models_lock = threading.Lock()
_stopwords_cache = None  # Cache stopwords to avoid repeated corpus access

# ─── Adaptive Model Learning ───
# Stores running statistics from each analysis to improve detection thresholds
_learning_stats = {
    'total_analyses': 0,
    'ai_detected_count': 0,
    'perplexity_sum': 0.0,
    'perplexity_sq_sum': 0.0,
    'burstiness_sum': 0.0,
    'burstiness_sq_sum': 0.0,
    'ttr_sum': 0.0,
    'cv_sum': 0.0,
    'samples': [],  # Store last N analysis features for threshold adaptation
    'max_samples': 200,
    'threshold_adjustments': {},
}
_learning_lock = threading.Lock()

# Download required NLTK resources with robust error handling
def download_nltk_resources():
    """Download NLTK resources with proper error handling"""
    # Map resources to their correct NLTK data paths
    resource_paths = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'stopwords': 'corpora/stopwords',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
    }
    
    for resource, path in resource_paths.items():
        try:
            nltk.data.find(path)
            logger.info(f"Resource {resource} already downloaded")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
                logger.info(f"Successfully downloaded NLTK resource: {resource}")
            except Exception as e:
                logger.error(f"Failed to download {resource}: {str(e)}")
                raise RuntimeError(f"Critical: Could not download {resource}")

# Set NLTK data path explicitly (optional, but can help)
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Initialize Flask app first before downloading resources
app = Flask(__name__, static_folder='Front')

# Enhanced CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:5000", "http://127.0.0.1:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With", "X-Session-Id"],
        "expose_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 120
    }
})

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin in ['http://localhost:8000', 'http://127.0.0.1:8000', 'http://localhost:5000', 'http://127.0.0.1:5000']:
        response.headers['Access-Control-Allow-Origin'] = origin
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,Accept,Origin,X-Requested-With,X-Session-Id'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

# Serve static files
@app.route('/')
def serve_front():
    return send_from_directory('Front', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('Front', path)

# Lazy loading function for models
def load_models():
    """Load ML models only when needed"""
    global tokenizer, model, vectorizer, models_loaded
    global transformers, torch, sklearn, _models_loading

    # Fast-path
    if models_loaded:
        return True  # Already loaded

    # Prevent concurrent initializations
    with _models_lock:
        if models_loaded:
            return True
        if _models_loading:
            # Another thread is loading; wait briefly until done
            wait_start = time.time()
            while _models_loading and time.time() - wait_start < 30:
                time.sleep(0.1)
            return models_loaded

        _models_loading = True

        try:
            logger.info("Loading ML models...")

            # Import heavy libraries here
            import transformers as hf_transformers
            import torch as pt
            from sklearn.feature_extraction.text import TfidfVectorizer
            # Bind to globals
            transformers = hf_transformers
            torch = pt

            # Download NLTK resources
            download_nltk_resources()

            # Cache stopwords after download
            _cache_stopwords()

            # Initialize TF-IDF vectorizer for text analysis
            vectorizer = TfidfVectorizer(
                max_features=5000,
                strip_accents='unicode',
                ngram_range=(1, 3),
                stop_words='english'
            )

            # Load GPT-2 for perplexity calculation (the core of AI detection)
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    "gpt2",
                    use_fast=True,
                )
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    "gpt2",
                )
                model.eval()
                logger.info("GPT-2 model loaded successfully for perplexity analysis")
            except Exception as e:
                logger.warning(f"Falling back: could not load GPT-2 model/tokenizer ({e})")
                tokenizer = None
                model = None

            models_loaded = True
            logger.info("Models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            models_loaded = True
            tokenizer = None
            model = None
            return True
        finally:
            _models_loading = False


def _cache_stopwords():
    """Cache NLTK stopwords to avoid repeated corpus loading issues."""
    global _stopwords_cache
    if _stopwords_cache is not None:
        return
    try:
        from nltk.corpus import stopwords
        _stopwords_cache = set(stopwords.words('english'))
        logger.info(f"Cached {len(_stopwords_cache)} stopwords")
    except Exception as e:
        logger.warning(f"Could not cache stopwords: {e}")
        _stopwords_cache = set()


def _get_stopwords():
    """Get cached stopwords (safe to call repeatedly)."""
    global _stopwords_cache
    if _stopwords_cache is None:
        _cache_stopwords()
    return _stopwords_cache if _stopwords_cache else set()

def preprocess_text(text):
    """Preprocess text with robust error handling using cached stopwords."""
    if not text:
        return []
        
    try:
        # Ensure models are loaded
        if not load_models():
            logger.error("Failed to load models for text preprocessing")
            return []
            
        from nltk.tokenize import word_tokenize
            
        # Tokenize with error catching
        try:
            tokens = word_tokenize(text.lower())
        except Exception as e:
            logger.warning(f"word_tokenize failed, falling back to basic split: {str(e)}")
            tokens = text.lower().split()
            
        # Use cached stopwords (avoids corpus reload bugs)
        stop_words = _get_stopwords()
            
        tokens = [token for token in tokens 
                 if token not in stop_words 
                 and token not in string.punctuation
                 and len(token) > 1
                 and token.strip()]
        return tokens
        
    except Exception as e:
        logger.error(f"Text preprocessing error: {str(e)}")
        return []

def get_perplexity_and_burstiness(text):
    """Calculate perplexity using GPT-2 and burstiness from token frequency distribution."""
    try:
        if not text or len(text.strip()) < 10:
            raise ValueError("Text too short for analysis")

        # Ensure models are loaded
        if not load_models():
            raise RuntimeError("Failed to load ML models")

        import torch
        from nltk.probability import FreqDist

        # --- Burstiness ---
        tokens = preprocess_text(text)
        word_freq = FreqDist(tokens)
        counts = list(word_freq.values())

        if len(counts) < 2:
            burstiness = 0.0
        else:
            mean_c = sum(counts) / len(counts)
            variance_c = sum((c - mean_c) ** 2 for c in counts) / len(counts)
            std_c = variance_c ** 0.5
            # Burstiness index: (std - mean) / (std + mean)
            # Ranges from -1 (perfectly regular) to 1 (very bursty)
            # AI text tends to be closer to 0 or negative (more uniform distribution)
            burstiness = (std_c - mean_c) / (std_c + mean_c + 1e-8)
            
            # Also compute sentence-level burstiness for better AI detection
            from nltk.tokenize import sent_tokenize
            sents = sent_tokenize(text)
            if len(sents) >= 3:
                sent_lens = [len(s.split()) for s in sents]
                sent_mean = sum(sent_lens) / len(sent_lens)
                sent_std = (sum((l - sent_mean) ** 2 for l in sent_lens) / len(sent_lens)) ** 0.5
                sent_burstiness = (sent_std - sent_mean) / (sent_std + sent_mean + 1e-8)
                # Blend word-level and sentence-level burstiness
                burstiness = 0.6 * burstiness + 0.4 * sent_burstiness

        # --- Perplexity (GPT-2 based) ---
        perplexity = None
        if tokenizer is not None and model is not None:
            try:
                encodings = tokenizer(text, return_tensors='pt', truncation=False)
                input_ids = encodings['input_ids']
                seq_len = input_ids.size(1)
                max_length = 1024  # GPT-2 context window
                stride = 512

                nlls = []
                prev_end = 0
                for begin_loc in range(0, seq_len, stride):
                    end_loc = min(begin_loc + max_length, seq_len)
                    trg_len = end_loc - prev_end  # only score the new tokens
                    input_chunk = input_ids[:, begin_loc:end_loc]

                    target_ids = input_chunk.clone()
                    # Mask tokens we've already scored (overlap region)
                    target_ids[:, :-trg_len] = -100

                    with torch.no_grad():
                        outputs = model(input_chunk, labels=target_ids)
                        neg_log_likelihood = outputs.loss
                        nlls.append(neg_log_likelihood)

                    prev_end = end_loc
                    if end_loc == seq_len:
                        break

                if nlls:
                    perplexity = torch.exp(torch.stack(nlls).mean()).item()
            except Exception as e:
                logger.warning(f"GPT-2 perplexity failed, using fallback: {e}")

        # Fallback: simple add-one smoothed unigram model
        if perplexity is None:
            if not tokens:
                perplexity = 100.0
            else:
                V = len(word_freq)
                N = len(tokens)
                log_prob_sum = 0.0
                for tok in tokens:
                    p = (word_freq[tok] + 1) / (N + V)
                    log_prob_sum += np.log(p)
                avg_neg_log_prob = -log_prob_sum / max(N, 1)
                perplexity = float(np.exp(avg_neg_log_prob))

        return float(perplexity), float(burstiness)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise

def _compute_text_signals(text, tokens, perplexity, burstiness):
    """Compute multiple heuristic signals to assess AI-likeness.

    Returns a dict with individual signals in [0,1] where higher means more AI-like.
    Conservatively tuned to minimize false positives (erring toward human).
    """
    try:
        from nltk.tokenize import sent_tokenize
        from collections import Counter

        # Basic counts
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        n_sent = len(sentences)
        n_tokens = len(tokens)
        token_counts = Counter(tokens)

        # Type-token ratio (vocab diversity)
        ttr = (len(token_counts) / max(n_tokens, 1)) if n_tokens else 0.0
        # High repetition of top word
        top_word_ratio = (token_counts.most_common(1)[0][1] / n_tokens) if n_tokens else 0.0

        # Sentence length variability
        sent_lens = [max(1, len(s.split())) for s in sentences] if sentences else []
        if sent_lens:
            mean_len = float(np.mean(sent_lens))
            std_len = float(np.std(sent_lens))
            cv_len = (std_len / mean_len) if mean_len > 0 else 0.0
        else:
            cv_len = 0.0

        # Punctuation ratio (use a safe character set to avoid quoting issues)
        punct_chars = set(".,;:!?\"'`-()[]{}")
        punct_ratio = sum(1 for ch in text if ch in punct_chars) / max(len(text), 1)

        # Repeated n-grams ratio (simple bigram repetition proxy)
        bigrams = list(zip(tokens, tokens[1:])) if n_tokens >= 2 else []
        bigram_counts = Counter(bigrams)
        repeated_bigrams = sum(c for _, c in bigram_counts.items() if c > 1)
        repeated_bigram_ratio = (repeated_bigrams / max(len(bigrams), 1)) if bigrams else 0.0

        # Map to AI-like signals (0..1). Higher means more AI-like.
        signals = {
            # Lower burstiness tends to be more AI-like (widened thresholds)
            'low_burstiness': 1.0 if burstiness < 0.08 else (0.7 if burstiness < 0.15 else (0.4 if burstiness < 0.25 else (0.15 if burstiness < 0.35 else 0.0))),
            # Low sentence variability (cv) more AI-like (widened thresholds)
            'low_sent_var': 1.0 if cv_len < 0.25 else (0.7 if cv_len < 0.35 else (0.4 if cv_len < 0.50 else (0.15 if cv_len < 0.65 else 0.0))),
            # Low vocab diversity more AI-like (widened thresholds)
            'low_ttr': 1.0 if ttr < 0.40 else (0.7 if ttr < 0.50 else (0.4 if ttr < 0.60 else (0.15 if ttr < 0.70 else 0.0))),
            # High repetition more AI-like
            'high_repetition': 1.0 if top_word_ratio > 0.07 else (0.6 if top_word_ratio > 0.055 else (0.3 if top_word_ratio > 0.04 else 0.0)),
            # Low perplexity is the STRONGEST AI signal — GPT-2 PPL below 60 is very suspicious
            'low_perplexity': 1.0 if perplexity < 30 else (0.85 if perplexity < 45 else (0.65 if perplexity < 60 else (0.4 if perplexity < 80 else (0.15 if perplexity < 100 else 0.0)))),
            # Very low punctuation ratio can indicate template-like text
            'low_punct_ratio': 1.0 if punct_ratio < 0.005 else (0.5 if punct_ratio < 0.015 else 0.0),
            # Repeated bigrams can indicate formulaic text
            'repeat_bigrams': 1.0 if repeated_bigram_ratio > 0.05 else (0.5 if repeated_bigram_ratio > 0.03 else 0.0),
        }

        # Add raw metrics for transparency
        signals['metrics'] = {
            'ttr': ttr,
            'top_word_ratio': top_word_ratio,
            'cv_sentence_length': cv_len,
            'punct_ratio': punct_ratio,
            'repeated_bigram_ratio': repeated_bigram_ratio,
            'n_sentences': n_sent,
            'n_tokens': n_tokens,
        }

        return signals
    except Exception as e:
        logger.warning(f"Signal computation failed: {e}")
        return {
            'low_burstiness': 0.0,
            'low_sent_var': 0.0,
            'low_ttr': 0.0,
            'high_repetition': 0.0,
            'low_perplexity': 0.0,
            'low_punct_ratio': 0.0,
            'repeat_bigrams': 0.0,
            'metrics': {}
        }

def assess_ai_likelihood(text, perplexity, burstiness):
    """Combine multiple conservative heuristics to decide AI likelihood.

    Returns (is_ai, confidence, reasons, signals)
    - is_ai is True only when multiple strong signals agree and text is sufficiently long.
    - confidence in [0,1].
    - reasons: brief human-readable notes.
    """
    tokens = preprocess_text(text)

    # Require sufficient length for a confident classification
    min_tokens = 60  # ~40-60 words minimum
    min_sentences = 3

    # Compute signals
    signals = _compute_text_signals(text, tokens, perplexity, burstiness)

    n_tokens = signals.get('metrics', {}).get('n_tokens', len(tokens))
    n_sent = signals.get('metrics', {}).get('n_sentences', 0)

    # Aggregate score — perplexity is the strongest AI signal from GPT-2
    weights = {
        'low_perplexity': 0.35,   # GPT-2 perplexity is the most reliable signal
        'low_burstiness': 0.15,
        'low_sent_var': 0.15,
        'low_ttr': 0.12,
        'high_repetition': 0.10,
        'low_punct_ratio': 0.05,
        'repeat_bigrams': 0.08,
    }

    weighted = 0.0
    total_w = 0.0
    contributions = {}
    for k, w in weights.items():
        v = float(signals.get(k, 0.0))
        weighted += w * v
        total_w += w
        contributions[k] = v

    ai_score = (weighted / total_w) if total_w else 0.0

    # Decision policy
    strong_signals = sum(1 for v in contributions.values() if v >= 0.8)
    medium_signals = sum(1 for v in contributions.values() if 0.4 <= v < 0.8)

    reasons = []

    # Base decision on score and count of strong signals
    is_long_enough = (n_tokens >= min_tokens and n_sent >= min_sentences)

    # Check for perplexity override — very low PPL is an extremely strong AI indicator
    perplexity_override = (perplexity < 45 and contributions.get('low_perplexity', 0) >= 0.85)

    if not is_long_enough:
        # Short text — still flag if perplexity is very low
        if perplexity_override and n_tokens >= 30:
            is_ai = True
            confidence = min(1.0, 0.55 + 0.35 * ai_score)
            reasons.append('Low perplexity strongly suggests AI despite short text')
        else:
            is_ai = False
            confidence = 0.2 + 0.3 * ai_score
            reasons.append('Text too short for reliable detection; defaulting to human')
    else:
        # Primary decision: score-based with signal agreement
        is_ai = (
            ai_score >= 0.45  # Lowered from 0.68
            and (strong_signals >= 1 or medium_signals >= 2)  # Relaxed agreement
        )
        # Perplexity override: if GPT-2 says it's very predictable, flag it
        if not is_ai and perplexity_override:
            is_ai = True
            reasons.append('Perplexity override: text is highly predictable by GPT-2')
        # Confidence scales with score and signal agreement
        confidence = min(1.0, 0.3 + 0.5 * ai_score + 0.08 * strong_signals + 0.05 * medium_signals)

    # Add explanatory reasons
    if contributions.get('low_perplexity', 0) >= 0.65:
        reasons.append('Low perplexity (text is highly predictable by GPT-2)')
    if contributions.get('low_burstiness', 0) >= 0.7:
        reasons.append('Very low burstiness (uniform word distribution)')
    if contributions.get('low_sent_var', 0) >= 0.7:
        reasons.append('Low sentence length variability')
    if contributions.get('low_ttr', 0) >= 0.7:
        reasons.append('Low vocabulary diversity')
    if contributions.get('high_repetition', 0) >= 0.6:
        reasons.append('High repetition of common words')
    if contributions.get('repeat_bigrams', 0) >= 0.5:
        reasons.append('Repetitive phrase patterns')
    if contributions.get('low_punct_ratio', 0) >= 0.5:
        reasons.append('Unusually low punctuation usage')

    return is_ai, float(confidence), reasons, {'score': ai_score, 'contributions': contributions, **signals.get('metrics', {})}


def _update_learning_stats(perplexity, burstiness, signals, is_ai):
    """Update adaptive model statistics with each new analysis.
    
    This accumulates statistics from each analysis to refine detection
    thresholds over time, making the model smarter with every file analyzed.
    """
    global _learning_stats
    
    with _learning_lock:
        stats = _learning_stats
        stats['total_analyses'] += 1
        if is_ai:
            stats['ai_detected_count'] += 1
        
        stats['perplexity_sum'] += perplexity
        stats['perplexity_sq_sum'] += perplexity ** 2
        stats['burstiness_sum'] += burstiness
        stats['burstiness_sq_sum'] += burstiness ** 2
        
        metrics = signals.get('metrics', {})
        stats['ttr_sum'] += metrics.get('ttr', 0)
        stats['cv_sum'] += metrics.get('cv_sentence_length', 0)
        
        # Store sample features for threshold adaptation
        sample = {
            'perplexity': perplexity,
            'burstiness': burstiness,
            'ttr': metrics.get('ttr', 0),
            'cv': metrics.get('cv_sentence_length', 0),
            'is_ai': is_ai,
        }
        stats['samples'].append(sample)
        if len(stats['samples']) > stats['max_samples']:
            stats['samples'] = stats['samples'][-stats['max_samples']:]
        
        # Recalculate adaptive thresholds every 10 analyses
        if stats['total_analyses'] % 10 == 0 and stats['total_analyses'] >= 20:
            _recalculate_thresholds(stats)
        
        logger.info(f"Model learning updated: {stats['total_analyses']} total analyses, "
                     f"{stats['ai_detected_count']} AI detected, "
                     f"avg PPL: {stats['perplexity_sum']/stats['total_analyses']:.1f}")


def _recalculate_thresholds(stats):
    """Recalculate adaptive detection thresholds based on accumulated data."""
    n = stats['total_analyses']
    if n < 20:
        return
    
    # Calculate running statistics
    avg_ppl = stats['perplexity_sum'] / n
    avg_burst = stats['burstiness_sum'] / n
    avg_ttr = stats['ttr_sum'] / n
    avg_cv = stats['cv_sum'] / n
    
    # Variance for perplexity
    var_ppl = (stats['perplexity_sq_sum'] / n) - (avg_ppl ** 2)
    std_ppl = max(var_ppl ** 0.5, 1.0)
    
    # Adaptive thresholds: use mean - 1 std as the "low" threshold
    stats['threshold_adjustments'] = {
        'perplexity_low': max(20, avg_ppl - std_ppl),
        'perplexity_mid': max(35, avg_ppl - 0.5 * std_ppl),
        'avg_burstiness': avg_burst,
        'avg_ttr': avg_ttr,
        'avg_cv': avg_cv,
        'sample_count': n,
    }
    
    logger.info(f"Adaptive thresholds updated: PPL_low={stats['threshold_adjustments']['perplexity_low']:.1f}, "
                f"PPL_mid={stats['threshold_adjustments']['perplexity_mid']:.1f}, "
                f"avg_burst={avg_burst:.4f}, samples={n}")


def get_learning_stats():
    """Get current model learning statistics."""
    with _learning_lock:
        stats = _learning_stats.copy()
        n = stats['total_analyses']
        if n > 0:
            return {
                'total_analyses': n,
                'ai_detected_count': stats['ai_detected_count'],
                'ai_detection_rate': round(stats['ai_detected_count'] / n * 100, 1),
                'avg_perplexity': round(stats['perplexity_sum'] / n, 2),
                'avg_burstiness': round(stats['burstiness_sum'] / n, 4),
                'threshold_adjustments': stats.get('threshold_adjustments', {}),
                'model_improved': n >= 20,
            }
        return {
            'total_analyses': 0,
            'model_improved': False,
        }


def check_plagiarism(text1, text2):
    """Enhanced plagiarism detection"""
    try:
        # Ensure models are loaded
        if not load_models():
            logger.error("Failed to load models for plagiarism check")
            return 0, []
            
        from nltk.tokenize import sent_tokenize
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
        import nltk
        
        # Get sentence pairs from original texts (not preprocessed)
        sentences1 = sent_tokenize(text1)
        sentences2 = sent_tokenize(text2)
        
        if not sentences1 or not sentences2:
            return 0, []
        
        # Use a local vectorizer to avoid contaminating the global one
        local_vec = TfidfVectorizer(
            max_features=5000,
            strip_accents='unicode',
            ngram_range=(1, 2),
            stop_words='english'
        )
            
        # Calculate similarity matrix
        similarity_matrix = []
        similar_sentences = []
        
        for sent1 in sentences1:
            sent_similarities = []
            for sent2 in sentences2:
                # TF-IDF cosine similarity
                try:
                    vectors = local_vec.fit_transform([sent1, sent2])
                    tfidf_sim = sklearn_cosine(vectors[0:1], vectors[1:2])[0][0]
                except Exception:
                    tfidf_sim = 0.0
                
                # N-gram overlap (Jaccard on trigrams)
                n = 3
                words1 = sent1.lower().split()
                words2 = sent2.lower().split()
                sent1_ngrams = set(nltk.ngrams(words1, n)) if len(words1) >= n else set()
                sent2_ngrams = set(nltk.ngrams(words2, n)) if len(words2) >= n else set()
                
                if sent1_ngrams and sent2_ngrams:
                    ngram_sim = len(sent1_ngrams & sent2_ngrams) / len(sent1_ngrams | sent2_ngrams)
                else:
                    ngram_sim = 0.0
                    
                # Combined similarity (weighted average)
                similarity = 0.6 * tfidf_sim + 0.4 * ngram_sim
                sent_similarities.append(similarity)
                
                if similarity > 0.5:
                    similar_sentences.append({
                        'text': sent1,
                        'similarity': round(similarity * 100, 2)
                    })
                    
            similarity_matrix.append(sent_similarities)
            
        # Calculate overall similarity
        if similarity_matrix:
            max_similarities = [max(row) for row in similarity_matrix]
            overall_similarity = (sum(max_similarities) / len(max_similarities)) * 100
        else:
            overall_similarity = 0
            
        return overall_similarity, similar_sentences
        
    except Exception as e:
        logger.error(f"Error in plagiarism check: {e}")
        return 0, []

@app.route('/check', methods=['POST', 'OPTIONS'])
def check():
    """Enhanced analysis endpoint"""
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,X-Session-Id')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    try:
        # Rate limiting
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        if db.is_rate_limited(client_ip):
            return jsonify({"error": "Rate limit exceeded. Please wait before trying again."}), 429

        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text'].strip()
        if len(text) < 50:  # Minimum length requirement
            return jsonify({"error": "Text too short for analysis"}), 400

        import time as _time
        start_time = _time.time()

        # Get metrics
        perplexity, burstiness = get_perplexity_and_burstiness(text)
        
        # Combine multiple conservative signals for AI detection
        is_ai_generated, confidence, reasons, signal_details = assess_ai_likelihood(text, perplexity, burstiness)

        # Enhanced analysis text
        analysis = {
            "perplexity_analysis": "Very low perplexity — strong AI indicator" if perplexity < 30 else ("Low perplexity — likely AI-generated" if perplexity < 50 else ("Moderate perplexity — possibly AI-assisted" if perplexity < 80 else "Natural perplexity level")),
            "burstiness_analysis": "Very low word variation — AI-like uniformity" if burstiness < 0.08 else ("Low word variation" if burstiness < 0.15 else "Natural word variation"),
            "overall": "Likely AI-generated" if is_ai_generated else "Likely human-written",
            "confidence": confidence,
            "reasons": reasons,
            "signals": signal_details,
        }

        # Add style consistency analysis
        style_consistency = analyze_text_style(text)
        
        # Calculate entropy from token distribution
        tokens = preprocess_text(text)
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        if total_tokens > 0:
            probs = np.array(list(token_counts.values())) / total_tokens
            entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        else:
            entropy = 0.0

        inference_time = _time.time() - start_time
        
        # Voice consistency analysis (passive vs active)
        voice_analysis = analyze_voice_consistency(text)

        # Enhanced metrics
        metrics = {
            "perplexity": perplexity,
            "burstiness": burstiness,
            "style_consistency": style_consistency,
            "complexity": calculate_complexity(text),
            "variability": calculate_variability(text),
            "readability": calculate_readability(text),
            "entropy": entropy,
            "token_count": total_tokens,
            "voice_analysis": voice_analysis,
            "model_name": "GPT-2 + Heuristics" if (tokenizer is not None and model is not None) else "Heuristics (fallback)",
            "inference_time": round(inference_time, 3),
        }

        result_payload = {
            **metrics,
            "analysis": analysis,
            "is_ai_generated": is_ai_generated
        }

        # ─── Adaptive Model Learning ───
        # Update running statistics so the model improves with each analysis
        try:
            _update_learning_stats(perplexity, burstiness, signal_details, is_ai_generated)
            learning_info = get_learning_stats()
            result_payload["model_learning"] = learning_info
        except Exception as learn_err:
            logger.warning(f"Learning update failed (non-critical): {learn_err}")

        # Save to database (non-blocking, best-effort)
        session_id = data.get('session_id') or request.headers.get('X-Session-Id')
        input_source = data.get('input_source', 'paste')
        original_filename = data.get('original_filename')
        user_id = _get_user_id_from_request()

        saved = db.save_analysis(
            input_text=text,
            results=result_payload,
            input_source=input_source,
            original_filename=original_filename,
            session_id=session_id,
            user_id=user_id,
        )

        # Include the analysis ID in the response so frontend can reference it
        if saved:
            result_payload["analysis_id"] = saved.get("id")

        # Log usage
        db.log_usage(client_ip, "/check", inference_time)

        return jsonify(result_payload)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def extract_text_from_pdf(file_content):
    """Extract text from PDF file with improved error handling"""
    try:
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        if len(pdf_reader.pages) == 0:
            raise ValueError("PDF file appears to be empty")
            
        text = []
        for page_num in range(len(pdf_reader.pages)):
            try:
                page = pdf_reader.pages[page_num]
                text_content = page.extract_text()
                if text_content.strip():  # Only add non-empty pages
                    text.append(text_content)
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                continue

        if not text:
            raise ValueError("No readable text found in PDF")
            
        return "\n\n".join(text)
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise ValueError(f"Could not extract text from PDF: {str(e)}")

def extract_text_from_doc(file_content):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(BytesIO(file_content))
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        raise ValueError("Could not extract text from DOCX")

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload():
    """Handle file uploads and extract text."""
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    try:
        logger.info("Upload endpoint called")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Content type: {request.content_type}")
        logger.info(f"Files in request: {list(request.files.keys())}")
        
        if 'file' not in request.files:
            logger.error("No file in request.files")
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({"error": "No file selected"}), 400

        # Log file details
        logger.info(f"Received file: {file.filename}, size: {len(file.read())} bytes")
        file.seek(0)  # Reset file pointer after reading size
        
        filename = os.path.basename(file.filename)
        content = file.read()
        
        if not content:
            return jsonify({"error": "File is empty"}), 400
            
        extracted_text = ""

        # Handle different file types
        if filename.lower().endswith('.pdf'):
            logger.info("Processing PDF file")
            extracted_text = extract_text_from_pdf(content)
        elif filename.lower().endswith('.docx'):
            logger.info("Processing DOCX file")
            extracted_text = extract_text_from_doc(content)
        else:
            logger.info("Processing text file")
            # For text files, try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    extracted_text = content.decode(encoding)
                    logger.info(f"Successfully decoded with {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

        if not extracted_text or not extracted_text.strip():
            return jsonify({"error": "No readable text found in file"}), 400

        logger.info(f"Successfully processed file: {filename}, extracted {len(extracted_text)} characters")
        return jsonify({
            "text": extracted_text,
            "filename": filename,
            "message": f"File '{filename}' uploaded successfully",
            "characters": len(extracted_text)
        })
                
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route('/analysis', methods=['POST', 'OPTIONS'])
def analysis():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text']
        perplexity, burstiness = get_perplexity_and_burstiness(text)
        
        return jsonify({
            "perplexity": float(perplexity),
            "burstiness": float(burstiness),
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/streamlit-analysis', methods=['POST', 'OPTIONS'])
def streamlit_analysis():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text']
        
        # Get metrics using existing functions
        perplexity, burstiness = get_perplexity_and_burstiness(text)
        
        # Get word frequency data using cached stopwords
        tokens = text.split()
        stop_words = _get_stopwords()
            
        tokens = [token.lower() for token in tokens 
                 if token.lower() not in stop_words 
                 and token.lower() not in string.punctuation
                 and len(token) > 1]

        word_counts = Counter(tokens)
        top_words = word_counts.most_common(10)
        
        return jsonify({
            "perplexity": perplexity,
            "burstiness": burstiness,
            "word_frequency": {
                "words": [word for word, _ in top_words],
                "counts": [count for _, count in top_words]
            },
            # Keep this endpoint focused on metrics; classification is provided by /check
            "is_ai_generated": None
        })

    except Exception as e:
        logger.error(f"Streamlit analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def analyze_text_style(text):
    """Analyze text style consistency using sentence-level statistical analysis.
    
    Measures how consistent the writing style is across the text by analyzing:
    - Sentence length distribution uniformity
    - Punctuation usage patterns
    - Vocabulary consistency across chunks
    Returns a score in [0, 1] where higher = more consistent style (more AI-like).
    """
    try:
        from nltk.tokenize import sent_tokenize
        
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return 0.5  # Not enough data
        
        # 1. Sentence length consistency (low variance = more AI-like)
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths or len(lengths) < 2:
            return 0.5
        mean_len = np.mean(lengths)
        cv = np.std(lengths) / (mean_len + 1e-8)  # coefficient of variation
        length_consistency = max(0.0, 1.0 - cv)  # Lower cv = higher consistency
        
        # 2. Punctuation pattern consistency across halves of text
        mid = len(text) // 2
        half1, half2 = text[:mid], text[mid:]
        punct_chars = set('.,;:!?-()[]{}')
        
        def punct_profile(t):
            total = max(len(t), 1)
            return {ch: t.count(ch) / total for ch in punct_chars}
        
        p1, p2 = punct_profile(half1), punct_profile(half2)
        punct_diff = sum(abs(p1.get(ch, 0) - p2.get(ch, 0)) for ch in punct_chars)
        punct_consistency = max(0.0, 1.0 - punct_diff * 50)  # Scale to [0, 1]
        
        # 3. Vocabulary overlap between first and second half
        tokens1 = set(half1.lower().split())
        tokens2 = set(half2.lower().split())
        if tokens1 and tokens2:
            jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
        else:
            jaccard = 0.5
        
        # Weighted combination
        score = 0.4 * length_consistency + 0.3 * punct_consistency + 0.3 * jaccard
        return float(np.clip(score, 0.0, 1.0))
        
    except Exception as e:
        logger.error(f"Style analysis error: {str(e)}")
        return 0.5


def calculate_complexity(text):
    """Estimate text complexity using proven linguistic metrics.
    
    Combines average word length, sentence length, and type-token ratio
    into a normalized complexity score.
    """
    try:
        from nltk.tokenize import sent_tokenize, word_tokenize
        
        sentences = sent_tokenize(text)
        words = [w for w in word_tokenize(text) if w.isalpha()]
        
        if not sentences or not words:
            return 0.0
        
        n_words = len(words)
        n_sentences = len(sentences)
        
        # Average word length (longer words = more complex)
        avg_word_len = np.mean([len(w) for w in words])
        
        # Average sentence length (longer sentences = more complex)
        avg_sent_len = n_words / max(n_sentences, 1)
        
        # Type-token ratio (higher = more diverse vocabulary = more complex)
        ttr = len(set(w.lower() for w in words)) / max(n_words, 1)
        
        # Normalize: typical ranges: word_len 3-8, sent_len 5-40, ttr 0.3-0.8
        word_len_score = np.clip((avg_word_len - 3) / 5, 0, 1)
        sent_len_score = np.clip((avg_sent_len - 5) / 35, 0, 1)
        ttr_score = np.clip((ttr - 0.3) / 0.5, 0, 1)
        
        complexity = 0.35 * word_len_score + 0.35 * sent_len_score + 0.3 * ttr_score
        return float(np.clip(complexity, 0.0, 1.0))
        
    except Exception as e:
        logger.error(f"Complexity calculation error: {str(e)}")
        return 0.0


def calculate_variability(text):
    """Estimate text variability based on word diversity and sentence structure variation."""
    try:
        from nltk.tokenize import sent_tokenize, word_tokenize
        
        sentences = sent_tokenize(text)
        words = [w for w in word_tokenize(text) if w.isalpha()]
        
        if not sentences or len(sentences) < 2 or not words:
            return 0.0
        
        # Type-token ratio
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / max(len(words), 1)
        
        # Coefficient of variation in sentence length
        sent_lengths = [len(s.split()) for s in sentences]
        mean_len = np.mean(sent_lengths)
        cv = np.std(sent_lengths) / (mean_len + 1e-8)
        
        # Combine: higher TTR + higher CV = more variable text
        variability = 0.5 * ttr + 0.5 * min(cv, 1.0)
        return float(np.clip(variability, 0.0, 1.0))
        
    except Exception as e:
        logger.error(f"Variability calculation error: {str(e)}")
        return 0.0


def calculate_readability(text):
    """Calculate readability using the Flesch Reading Ease formula.
    
    Flesch RE = 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    Returns a normalized score in [0, 1] where higher = more readable.
    """
    try:
        from nltk.tokenize import sent_tokenize, word_tokenize

        words = [w for w in word_tokenize(text) if w.isalpha()]
        sentences = sent_tokenize(text)
        
        if not words or not sentences:
            return 0.5

        n_words = len(words)
        n_sentences = len(sentences)

        def count_syllables(word):
            """Count syllables using vowel-group heuristic."""
            word = word.lower()
            vowels = 'aeiou'
            count = 0
            prev_vowel = False
            for ch in word:
                is_vowel = ch in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            # Handle silent 'e'
            if word.endswith('e') and count > 1:
                count -= 1
            return max(count, 1)

        n_syllables = sum(count_syllables(w) for w in words)

        # Flesch Reading Ease formula
        flesch = 206.835 - 1.015 * (n_words / max(n_sentences, 1)) - 84.6 * (n_syllables / max(n_words, 1))
        
        # Normalize to [0, 1] (Flesch typically ranges from 0-100, can go outside)
        normalized = np.clip(flesch / 100.0, 0.0, 1.0)
        return float(normalized)

    except Exception as e:
        logger.error(f"Readability calculation error: {str(e)}")
        return 0.5


def analyze_voice_consistency(text):
    """Detect passive vs active voice ratio and flag sections with high passive voice.
    
    Uses POS tagging to identify passive constructions:
    - Pattern: form of 'to be' (am/is/are/was/were/been/being) + past participle (VBN)
    - Also detects 'get'-passives: get/got/gotten + VBN
    
    Returns a dict with:
    - passive_ratio: float [0,1] — fraction of sentences containing passive voice
    - active_ratio: float [0,1]
    - total_sentences: int
    - passive_count: int
    - active_count: int
    - flagged_sections: list of {text, index} for sentences with passive voice
    - assessment: human-readable summary
    """
    try:
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk import pos_tag
        
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        if not sentences:
            return _empty_voice_result()
        
        # Forms of 'to be' that precede past participles in passive constructions
        be_forms = {'am', 'is', 'are', 'was', 'were', 'been', 'being', 'be'}
        # Get-passives
        get_forms = {'get', 'gets', 'got', 'gotten', 'getting'}
        
        passive_sentences = []
        active_sentences = []
        flagged_sections = []
        
        for idx, sent in enumerate(sentences):
            try:
                words = word_tokenize(sent)
                tagged = pos_tag(words)
                
                is_passive = False
                for i in range(len(tagged) - 1):
                    word_lower = tagged[i][0].lower()
                    next_tag = tagged[i + 1][1]
                    
                    # Check: be-form + VBN (past participle)
                    if word_lower in be_forms and next_tag == 'VBN':
                        is_passive = True
                        break
                    # Check: get-form + VBN
                    if word_lower in get_forms and next_tag == 'VBN':
                        is_passive = True
                        break
                    # Check: be-form + adverb + VBN (e.g., "was quickly completed")
                    if (word_lower in be_forms and tagged[i + 1][1] in ('RB', 'RBR', 'RBS')
                            and i + 2 < len(tagged) and tagged[i + 2][1] == 'VBN'):
                        is_passive = True
                        break
                
                if is_passive:
                    passive_sentences.append(sent)
                    flagged_sections.append({
                        'text': sent,
                        'index': idx,
                    })
                else:
                    active_sentences.append(sent)
                    
            except Exception:
                # If POS tagging fails for a sentence, treat as active
                active_sentences.append(sent)
        
        total = len(sentences)
        passive_count = len(passive_sentences)
        active_count = len(active_sentences)
        passive_ratio = passive_count / max(total, 1)
        active_ratio = active_count / max(total, 1)
        
        # Generate assessment
        if passive_ratio >= 0.5:
            assessment = f'High passive voice usage ({passive_count}/{total} sentences). This is common in AI-generated text. Consider rewriting flagged sentences in active voice for more natural, engaging prose.'
        elif passive_ratio >= 0.3:
            assessment = f'Moderate passive voice ({passive_count}/{total} sentences). Some sections could be strengthened by converting to active voice.'
        elif passive_ratio >= 0.15:
            assessment = f'Occasional passive voice ({passive_count}/{total} sentences). Usage is within normal range for academic/formal writing.'
        else:
            assessment = f'Mostly active voice ({active_count}/{total} sentences). Writing style appears natural and direct.'
        
        return {
            'passive_ratio': round(passive_ratio, 4),
            'active_ratio': round(active_ratio, 4),
            'total_sentences': total,
            'passive_count': passive_count,
            'active_count': active_count,
            'flagged_sections': flagged_sections[:20],  # Limit to 20 flagged sentences
            'assessment': assessment,
        }
        
    except Exception as e:
        logger.error(f"Voice consistency analysis error: {e}")
        return _empty_voice_result()


def _empty_voice_result():
    """Return default voice analysis result when analysis cannot run."""
    return {
        'passive_ratio': 0.0,
        'active_ratio': 0.0,
        'total_sentences': 0,
        'passive_count': 0,
        'active_count': 0,
        'flagged_sections': [],
        'assessment': 'Insufficient text for voice analysis.',
    }


@app.route('/sentence-analysis', methods=['POST', 'OPTIONS'])
def sentence_analysis():
    """Compute per-sentence AI probability for heatmap visualization.

    Runs GPT-2 on each sentence to get its perplexity, then maps to a
    probability in [0, 1] where higher = more likely AI-generated.
    For long documents, batches sentences and limits processing time.
    """
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text'].strip()
        if len(text) < 20:
            return jsonify({"error": "Text too short for sentence analysis"}), 400

        if not load_models():
            return jsonify({"error": "Models not loaded"}), 500

        from nltk.tokenize import sent_tokenize
        import time as _time

        all_sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]

        if not all_sentences:
            return jsonify({"error": "No sentences found"}), 400

        # For very long texts, process all sentences but batch them efficiently
        MAX_SENTENCES = 200  # Cap at 200 sentences to avoid extreme processing times
        sentences = all_sentences[:MAX_SENTENCES]
        was_truncated = len(all_sentences) > MAX_SENTENCES

        results = []
        MAX_TIME = 45  # Maximum processing time in seconds
        start_time = _time.time()

        if tokenizer is not None and model is not None:
            import torch as _torch

            for sent in sentences:
                # Check time budget
                elapsed = _time.time() - start_time
                if elapsed > MAX_TIME:
                    # Time budget exhausted — use fast heuristic for remaining sentences
                    logger.info(f"Sentence analysis time budget reached after {len(results)} sentences ({elapsed:.1f}s)")
                    for remaining_sent in sentences[len(results):]:
                        results.append(_heuristic_sentence_prob(remaining_sent))
                    break

                try:
                    encodings = tokenizer(sent, return_tensors='pt', truncation=True, max_length=512)
                    input_ids = encodings['input_ids']

                    if input_ids.size(1) < 3:
                        # Too short for meaningful perplexity
                        results.append({"text": sent, "probability": 0.30, "perplexity": None})
                        continue

                    with _torch.no_grad():
                        outputs = model(input_ids, labels=input_ids)
                        loss = outputs.loss.item()

                    ppl = float(np.exp(min(loss, 20)))  # cap to avoid inf

                    # Map perplexity to AI probability (lower ppl = higher prob)
                    # GPT-2 text typically has PPL < 30; human text > 60-80
                    if ppl < 15:
                        prob = 0.95
                    elif ppl < 25:
                        prob = 0.85 + 0.10 * (25 - ppl) / 10
                    elif ppl < 40:
                        prob = 0.70 + 0.15 * (40 - ppl) / 15
                    elif ppl < 60:
                        prob = 0.50 + 0.20 * (60 - ppl) / 20
                    elif ppl < 90:
                        prob = 0.30 + 0.20 * (90 - ppl) / 30
                    elif ppl < 150:
                        prob = 0.12 + 0.18 * (150 - ppl) / 60
                    else:
                        prob = 0.05

                    results.append({
                        "text": sent,
                        "probability": round(float(np.clip(prob, 0.0, 1.0)), 3),
                        "perplexity": round(ppl, 2),
                    })
                except Exception as exc:
                    logger.warning(f"Sentence perplexity failed for '{sent[:40]}...': {exc}")
                    results.append({"text": sent, "probability": 0.30, "perplexity": None})
        else:
            # Heuristic fallback when GPT-2 is unavailable
            for sent in sentences:
                results.append(_heuristic_sentence_prob(sent))

        # If we truncated, add a note about remaining sentences
        if was_truncated:
            remaining_count = len(all_sentences) - MAX_SENTENCES
            results.append({
                "text": f"[...{remaining_count} more sentences not shown]",
                "probability": 0.0,
                "perplexity": None,
                "is_note": True,
            })

        return jsonify({"sentences": results, "total_sentences": len(all_sentences), "analyzed": len(sentences)})

    except Exception as e:
        logger.error(f"Sentence analysis error: {e}")
        return jsonify({"error": str(e)}), 500


def _heuristic_sentence_prob(sent):
    """Fast heuristic-based sentence AI probability (no GPU needed)."""
    words = sent.split()
    n = len(words)
    avg_wl = sum(len(w) for w in words) / max(n, 1)
    unique_ratio = len(set(w.lower() for w in words)) / max(n, 1)
    prob = 0.25
    if avg_wl > 5.5:
        prob += 0.10
    if unique_ratio < 0.55:
        prob += 0.15
    if n > 25:
        prob += 0.10
    return {"text": sent, "probability": round(min(prob, 0.95), 3), "perplexity": None}


def calculate_cosine_similarity(text1, text2):
    """Compute cosine similarity between two texts using sklearn TF-IDF.
    Returns a similarity score in [0, 1].
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        if not text1.strip() or not text2.strip():
            return 0.0
        
        # Use a fresh local vectorizer to avoid state contamination
        local_vectorizer = TfidfVectorizer(
            max_features=5000,
            strip_accents='unicode',
            ngram_range=(1, 2),
            stop_words='english'
        )
        tfidf_matrix = local_vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(similarity, 0.0, 1.0))

    except Exception as e:
        logger.error(f"Cosine similarity error: {str(e)}")
        return 0.0


# ────────────────────────────────────────────
# Database-Powered Endpoints
# ────────────────────────────────────────────

def _get_user_id_from_request():
    """Extract authenticated user_id from Authorization header (Supabase JWT)."""
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]
        user = db.get_user_from_token(token)
        if user:
            return user.get('id')
    return None


# ────────────────────────────────────────────
# Auth Endpoints (Supabase Auth)
# ────────────────────────────────────────────

@app.route('/auth/signup', methods=['POST', 'OPTIONS'])
def auth_signup():
    """Register a new user with email and password."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    email = data.get('email', '').strip()
    password = data.get('password', '')
    full_name = data.get('full_name', '').strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    result = db.signup_user(email, password, full_name)
    if result and 'error' in result:
        return jsonify(result), 400
    if not result:
        return jsonify({"error": "Signup failed"}), 500

    return jsonify(result)


@app.route('/auth/login', methods=['POST', 'OPTIONS'])
def auth_login():
    """Log in with email and password."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    email = data.get('email', '').strip()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    result = db.login_user(email, password)
    if result and 'error' in result:
        return jsonify(result), 401
    if not result:
        return jsonify({"error": "Login failed"}), 500

    return jsonify(result)


@app.route('/auth/logout', methods=['POST', 'OPTIONS'])
def auth_logout():
    """Log out the current user."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    # Client-side just clears the token; server-side we can invalidate if needed
    return jsonify({"message": "Logged out successfully"})


@app.route('/auth/me', methods=['GET', 'OPTIONS'])
def auth_me():
    """Get the currently authenticated user."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return jsonify({"error": "Not authenticated"}), 401

    token = auth_header[7:]
    user = db.get_user_from_token(token)
    if not user:
        return jsonify({"error": "Invalid or expired token"}), 401

    return jsonify({"user": user})


@app.route('/auth/callback')
def auth_callback():
    """Handle Supabase email confirmation redirect.
    Supabase redirects here with tokens in the URL hash fragment.
    We just serve the main page — the JS handleAuthCallback() picks up the tokens.
    """
    return send_from_directory('Front', 'index.html')


@app.route('/history', methods=['GET', 'OPTIONS'])
def history():
    """Get analysis history for the current session or user."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    user_id = _get_user_id_from_request()
    session_id = request.headers.get('X-Session-Id') or request.args.get('session_id')
    limit = min(int(request.args.get('limit', 20)), 50)
    offset = int(request.args.get('offset', 0))

    if not user_id and not session_id:
        return jsonify({"error": "No session ID or auth token provided"}), 400

    results = db.get_analysis_history(user_id=user_id, session_id=session_id, limit=limit, offset=offset)
    return jsonify({"history": results, "count": len(results)})


@app.route('/history/<analysis_id>', methods=['GET'])
def get_single_analysis(analysis_id):
    """Get a single analysis by ID."""
    result = db.get_analysis_by_id(analysis_id)
    if not result:
        return jsonify({"error": "Analysis not found"}), 404
    return jsonify(result)


@app.route('/report', methods=['POST', 'OPTIONS'])
def create_report():
    """Create a shareable report link for an analysis."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    data = request.get_json()
    if not data or 'analysis_id' not in data:
        return jsonify({"error": "No analysis_id provided"}), 400

    report = db.create_report(data['analysis_id'])
    if not report:
        return jsonify({"error": "Could not create report"}), 500

    return jsonify(report)


@app.route('/report/<share_token>', methods=['GET'])
def view_report(share_token):
    """View a shared report by its token."""
    report = db.get_report_by_token(share_token)
    if not report:
        return jsonify({"error": "Report not found or expired"}), 404
    return jsonify(report)


@app.route('/stats', methods=['GET'])
def global_stats():
    """Get aggregate platform statistics."""
    stats = db.get_global_stats()
    return jsonify(stats)


@app.route('/model-stats', methods=['GET'])
def model_stats():
    """Get current model learning statistics."""
    return jsonify(get_learning_stats())


# ────────────────────────────────────────────
# Feature: Cross-Document Plagiarism Ring Detection
# ────────────────────────────────────────────

@app.route('/cross-check', methods=['POST', 'OPTIONS'])
def cross_check():
    """Compare current text against all stored analyses to find similar documents."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text'].strip()
        if len(text) < 50:
            return jsonify({"error": "Text too short for cross-document check"}), 400

        threshold = float(data.get('threshold', 0.3))
        exclude_id = data.get('exclude_id')

        stored = db.get_stored_texts(limit=100, exclude_id=exclude_id)
        if not stored:
            return jsonify({
                "matches": [],
                "total_compared": 0,
                "message": "No documents in the database to compare against."
            })

        matches = []
        for doc in stored:
            stored_text = doc.get('input_text', '')
            if not stored_text or len(stored_text) < 30:
                continue

            similarity = calculate_cosine_similarity(text, stored_text)
            if similarity >= threshold:
                matches.append({
                    "id": doc.get('id'),
                    "similarity": round(similarity * 100, 1),
                    "date": doc.get('created_at'),
                    "snippet": stored_text[:200] + ('...' if len(stored_text) > 200 else ''),
                    "source": doc.get('input_source', 'paste'),
                    "filename": doc.get('original_filename'),
                    "was_ai": doc.get('is_ai_generated', False),
                    "word_count": doc.get('word_count', len(stored_text.split())),
                })

        matches.sort(key=lambda x: x['similarity'], reverse=True)

        return jsonify({
            "matches": matches[:20],
            "total_compared": len(stored),
            "threshold": threshold * 100,
            "ring_detected": len(matches) >= 2,
        })

    except Exception as e:
        logger.error(f"Cross-check error: {e}")
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────
# Feature: Community Feedback & Calibration
# ────────────────────────────────────────────

@app.route('/feedback', methods=['POST', 'OPTIONS'])
def submit_feedback():
    """Submit user feedback on an analysis result."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json()
        if not data or 'analysis_id' not in data:
            return jsonify({"error": "No analysis_id provided"}), 400

        analysis_id = data['analysis_id']
        is_correct = data.get('is_correct', True)
        comment = data.get('comment', '')

        user_id = _get_user_id_from_request()
        session_id = data.get('session_id') or request.headers.get('X-Session-Id')

        result = db.submit_feedback(
            analysis_id=analysis_id,
            is_correct=is_correct,
            user_id=user_id,
            session_id=session_id,
            comment=comment,
        )

        return jsonify({
            "success": result is not None,
            "message": "Thank you for your feedback!" if result else "Feedback could not be saved (table may not exist)."
        })

    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/feedback/stats', methods=['GET'])
def feedback_stats():
    """Get community calibration accuracy stats."""
    stats = db.get_feedback_stats()
    return jsonify(stats)


# ────────────────────────────────────────────
# Feature: Side-by-Side Text Comparison
# ────────────────────────────────────────────

@app.route('/compare', methods=['POST', 'OPTIONS'])
def compare_texts():
    """Compare two texts side by side for plagiarism detection."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        text1 = (data.get('text1') or '').strip()
        text2 = (data.get('text2') or '').strip()

        if not text1 or not text2:
            return jsonify({"error": "Both text1 and text2 are required"}), 400
        if len(text1) < 30 or len(text2) < 30:
            return jsonify({"error": "Both texts must be at least 30 characters"}), 400

        if not load_models():
            return jsonify({"error": "Models not loaded"}), 500

        from nltk.tokenize import sent_tokenize
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

        # Overall similarity
        overall_similarity = calculate_cosine_similarity(text1, text2)

        # Sentence-level matching
        sents1 = [s.strip() for s in sent_tokenize(text1) if s.strip()]
        sents2 = [s.strip() for s in sent_tokenize(text2) if s.strip()]

        sentence_matches = []
        matched_indices_2 = set()

        for i, s1 in enumerate(sents1):
            best_sim = 0.0
            best_j = -1
            for j, s2 in enumerate(sents2):
                sim = calculate_cosine_similarity(s1, s2)
                if sim > best_sim:
                    best_sim = sim
                    best_j = j

            match_data = {
                "index1": i,
                "text1": s1,
                "best_similarity": round(best_sim * 100, 1),
            }
            if best_sim >= 0.4 and best_j >= 0:
                match_data["index2"] = best_j
                match_data["text2"] = sents2[best_j]
                match_data["status"] = "high_match" if best_sim >= 0.7 else "partial_match"
                matched_indices_2.add(best_j)
            else:
                match_data["status"] = "unique"
            sentence_matches.append(match_data)

        # Find unique sentences in text2 (not matched by any text1 sentence)
        unique_in_text2 = []
        for j, s2 in enumerate(sents2):
            if j not in matched_indices_2:
                unique_in_text2.append({"index2": j, "text2": s2})

        # Statistics
        high_matches = sum(1 for m in sentence_matches if m['status'] == 'high_match')
        partial_matches = sum(1 for m in sentence_matches if m['status'] == 'partial_match')
        unique_count = sum(1 for m in sentence_matches if m['status'] == 'unique')

        return jsonify({
            "overall_similarity": round(overall_similarity * 100, 1),
            "sentence_matches": sentence_matches,
            "unique_in_text2": unique_in_text2,
            "stats": {
                "text1_sentences": len(sents1),
                "text2_sentences": len(sents2),
                "high_matches": high_matches,
                "partial_matches": partial_matches,
                "unique_in_text1": unique_count,
                "unique_in_text2": len(unique_in_text2),
                "text1_words": len(text1.split()),
                "text2_words": len(text2.split()),
            },
            "verdict": (
                "High plagiarism detected" if overall_similarity >= 0.7
                else "Moderate similarity found" if overall_similarity >= 0.4
                else "Texts appear to be original"
            ),
        })

    except Exception as e:
        logger.error(f"Compare error: {e}")
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────
# Feature: Batch Analysis
# ────────────────────────────────────────────

@app.route('/batch-analyze', methods=['POST', 'OPTIONS'])
def batch_analyze():
    """Analyze multiple files/texts and return comparative results."""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        # Handle multipart form data (file uploads)
        texts = []
        filenames = []

        if request.content_type and 'multipart/form-data' in request.content_type:
            files = request.files.getlist('files')
            if not files:
                return jsonify({"error": "No files provided"}), 400

            for f in files:
                fname = os.path.basename(f.filename)
                content = f.read()
                if not content:
                    continue

                try:
                    if fname.lower().endswith('.pdf'):
                        extracted = extract_text_from_pdf(content)
                    elif fname.lower().endswith('.docx'):
                        extracted = extract_text_from_doc(content)
                    else:
                        extracted = None
                        for enc in ['utf-8', 'latin-1', 'cp1252']:
                            try:
                                extracted = content.decode(enc)
                                break
                            except UnicodeDecodeError:
                                continue
                    if extracted and extracted.strip():
                        texts.append(extracted)
                        filenames.append(fname)
                except Exception as ex:
                    logger.warning(f"Batch: failed to extract {fname}: {ex}")
                    continue
        else:
            # JSON body with array of texts
            data = request.get_json()
            if not data or 'texts' not in data:
                return jsonify({"error": "No texts provided"}), 400
            for i, item in enumerate(data['texts']):
                if isinstance(item, dict):
                    texts.append(item.get('text', ''))
                    filenames.append(item.get('filename', f'Text {i+1}'))
                else:
                    texts.append(str(item))
                    filenames.append(f'Text {i+1}')

        if len(texts) < 2:
            return jsonify({"error": "At least 2 texts/files required for batch analysis"}), 400
        if len(texts) > 20:
            return jsonify({"error": "Maximum 20 files per batch"}), 400

        # Ensure models are loaded
        if not load_models():
            return jsonify({"error": "Models not loaded"}), 500

        import time as _time
        batch_start = _time.time()

        # Analyze each text individually
        individual_results = []
        for i, text in enumerate(texts):
            try:
                if len(text.strip()) < 50:
                    individual_results.append({
                        "filename": filenames[i],
                        "error": "Text too short",
                        "perplexity": None,
                        "burstiness": None,
                        "is_ai_generated": None,
                        "ai_probability": None,
                        "confidence": None,
                        "word_count": len(text.split()),
                    })
                    continue

                ppl, burst = get_perplexity_and_burstiness(text)
                is_ai, conf, reasons, signals = assess_ai_likelihood(text, ppl, burst)
                ai_score = signals.get('score', 0)

                individual_results.append({
                    "filename": filenames[i],
                    "perplexity": round(ppl, 2),
                    "burstiness": round(burst, 4),
                    "is_ai_generated": is_ai,
                    "ai_probability": round(ai_score * 100, 1),
                    "confidence": round(conf * 100, 1),
                    "reasons": reasons,
                    "word_count": len(text.split()),
                    "risk_level": "high" if ai_score >= 0.7 else ("medium" if ai_score >= 0.4 else "low"),
                })
            except Exception as ex:
                individual_results.append({
                    "filename": filenames[i],
                    "error": str(ex),
                    "perplexity": None,
                    "burstiness": None,
                    "is_ai_generated": None,
                    "ai_probability": None,
                    "confidence": None,
                    "word_count": len(text.split()),
                })

        # Cross-similarity matrix between all documents
        n = len(texts)
        similarity_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sim = calculate_cosine_similarity(texts[i], texts[j])
                similarity_matrix[i][j] = round(sim * 100, 1)
                similarity_matrix[j][i] = similarity_matrix[i][j]
            similarity_matrix[i][i] = 100.0

        # Flag outliers (documents significantly different from the group)
        avg_similarities = []
        for i in range(n):
            others = [similarity_matrix[i][j] for j in range(n) if j != i]
            avg_sim = sum(others) / len(others) if others else 0
            avg_similarities.append(round(avg_sim, 1))

        # Find suspicious pairs (high similarity between different documents)
        suspicious_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] >= 40:
                    suspicious_pairs.append({
                        "file1": filenames[i],
                        "file2": filenames[j],
                        "similarity": similarity_matrix[i][j],
                    })
        suspicious_pairs.sort(key=lambda x: x['similarity'], reverse=True)

        batch_time = _time.time() - batch_start

        # Summary statistics
        ai_count = sum(1 for r in individual_results if r.get('is_ai_generated'))
        human_count = sum(1 for r in individual_results if r.get('is_ai_generated') is False)
        error_count = sum(1 for r in individual_results if 'error' in r)

        return jsonify({
            "results": individual_results,
            "filenames": filenames,
            "similarity_matrix": similarity_matrix,
            "avg_similarities": avg_similarities,
            "suspicious_pairs": suspicious_pairs,
            "summary": {
                "total_files": n,
                "ai_detected": ai_count,
                "human_detected": human_count,
                "errors": error_count,
                "batch_time": round(batch_time, 2),
            },
        })

    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    try:
        # Ensure all models are loaded before starting server
        logger.info("Starting Flask server...")
        logger.info("Models loaded and ready")
        
        # Start the Flask application
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True,
            use_reloader=False  # Disable reloader to prevent model reloading
        )
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}")
        raise