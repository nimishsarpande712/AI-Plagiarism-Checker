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

# Download required NLTK resources with robust error handling
def download_nltk_resources():
    """Download NLTK resources with proper error handling"""
    # Map resources to their correct NLTK data paths
    resource_paths = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'stopwords': 'corpora/stopwords',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
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
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
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
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,Accept,Origin,X-Requested-With'
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
            # Fano factor-based burstiness: maps to (-1, 1), negative = regular, positive = bursty
            fano = variance_c / (mean_c + 1e-8)
            burstiness = (fano - 1) / (fano + 1)

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
            # Lower burstiness tends to be more AI-like
            'low_burstiness': 1.0 if burstiness < 0.04 else (0.6 if burstiness < 0.08 else (0.3 if burstiness < 0.12 else 0.0)),
            # Low sentence variability (cv) more AI-like
            'low_sent_var': 1.0 if cv_len < 0.22 else (0.6 if cv_len < 0.3 else (0.3 if cv_len < 0.4 else 0.0)),
            # Low vocab diversity more AI-like
            'low_ttr': 1.0 if ttr < 0.35 else (0.6 if ttr < 0.45 else (0.3 if ttr < 0.55 else 0.0)),
            # High repetition more AI-like
            'high_repetition': 1.0 if top_word_ratio > 0.08 else (0.6 if top_word_ratio > 0.065 else (0.3 if top_word_ratio > 0.055 else 0.0)),
            # Extremely low perplexity can be AI-like; use conservative thresholds
            'low_perplexity': 1.0 if perplexity < 40 else (0.6 if perplexity < 55 else (0.3 if perplexity < 70 else 0.0)),
            # Very low punctuation ratio can indicate template-like text
            'low_punct_ratio': 1.0 if punct_ratio < 0.005 else (0.5 if punct_ratio < 0.01 else 0.0),
            # Repeated bigrams can indicate formulaic text
            'repeat_bigrams': 1.0 if repeated_bigram_ratio > 0.06 else (0.5 if repeated_bigram_ratio > 0.04 else 0.0),
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
    min_tokens = 120  # ~80-120 words minimum
    min_sentences = 4

    # Compute signals
    signals = _compute_text_signals(text, tokens, perplexity, burstiness)

    n_tokens = signals.get('metrics', {}).get('n_tokens', len(tokens))
    n_sent = signals.get('metrics', {}).get('n_sentences', 0)

    # Aggregate score with conservative weights
    weights = {
        'low_burstiness': 0.22,
        'low_sent_var': 0.22,
        'low_ttr': 0.18,
        'high_repetition': 0.18,
        'low_perplexity': 0.12,
        'low_punct_ratio': 0.04,
        'repeat_bigrams': 0.04,
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
    strong_signals = sum(1 for v in contributions.values() if v >= 1.0)
    medium_signals = sum(1 for v in contributions.values() if 0.5 <= v < 1.0)

    reasons = []

    # Base decision on score and count of strong signals
    is_long_enough = (n_tokens >= min_tokens and n_sent >= min_sentences)

    if not is_long_enough:
        # Too short for reliable decision; be conservative (lean human)
        is_ai = False
        confidence = 0.2 + 0.2 * ai_score  # Low confidence
        reasons.append('Text too short for reliable detection; defaulting to human')
    else:
        # Require both a reasonably high score and multiple strong/medium agreements
        is_ai = (ai_score >= 0.68 and (strong_signals >= 2 or (strong_signals >= 1 and medium_signals >= 2)))
        # Confidence increases with score and agreement
        confidence = min(1.0, 0.5 + 0.4 * ai_score + 0.05 * strong_signals + 0.03 * medium_signals)

    # Add explanatory reasons
    if contributions.get('low_burstiness', 0) >= 1.0:
        reasons.append('Very low burstiness')
    if contributions.get('low_sent_var', 0) >= 1.0:
        reasons.append('Low sentence length variability')
    if contributions.get('low_ttr', 0) >= 1.0:
        reasons.append('Low vocabulary diversity')
    if contributions.get('high_repetition', 0) >= 1.0:
        reasons.append('High repetition of common words')
    if contributions.get('low_perplexity', 0) >= 1.0:
        reasons.append('Very low perplexity')

    return is_ai, float(confidence), reasons, {'score': ai_score, 'contributions': contributions, **signals.get('metrics', {})}

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
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    try:
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
            "perplexity_analysis": "Very low perplexity can indicate AI" if perplexity < 40 else ("Somewhat low perplexity" if perplexity < 55 else "Natural perplexity level"),
            "burstiness_analysis": "Very low word variation" if burstiness < 0.04 else ("Somewhat low word variation" if burstiness < 0.08 else "Natural word variation"),
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
            "model_name": "GPT-2 + Heuristics" if (tokenizer is not None and model is not None) else "Heuristics (fallback)",
            "inference_time": round(inference_time, 3),
        }
        
        return jsonify({
            **metrics,
            "analysis": analysis,
            "is_ai_generated": is_ai_generated
        })

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