# AI & Plagiarism Detector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated web application designed to analyze text for potential AI generation and plagiarism. This tool provides detailed metrics and a clear verdict, making it useful for educators, writers, and content creators.




---

## ✨ Features

- **Advanced AI Text Detection**: Utilizes a multi-faceted approach to distinguish between human-written and AI-generated content.
- **Dual-Metric Analysis**: Core detection is based on **Perplexity** (predictability of text) and **Burstiness** (variation in sentence structure).
- **Heuristic Signal Processing**: Goes beyond basic metrics to analyze:
    - Vocabulary Diversity (Type-Token Ratio)
    - Sentence Length Variability
    - Repetition of Common Words
    - Punctuation Patterns
    - Repetitive N-grams
- **File Upload Capability**: Analyze content directly from `.pdf`, `.docx`, and `.txt` files.
- **Plagiarism Checking**: Compares text against another source using TF-IDF and n-gram overlap to detect similar sentences.
- **Detailed Analytics**: Provides a suite of metrics for in-depth analysis:
    - **Style Consistency**: Measures if the writing style is consistent throughout the text.
    - **Complexity**: Estimates the complexity of the text based on sentence length and word rarity.
    - **Variability**: Assesses the diversity of words and sentence structures.
- **Clear & Concise UI**: A clean, user-friendly interface for easy text input, file uploading, and results visualization.
- **RESTful API**: A Flask-based backend that exposes endpoints for analysis, which can be integrated into other services.

---

## ⚙️ How It Works

The application combines established NLP techniques with a multi-signal heuristic engine to provide a robust analysis.

### AI Detection Engine

The core of the AI detection lies in its ability to spot the subtle, non-human patterns in text.

1.  **Model Loading**: To ensure a fast startup, heavy machine learning models (like GPT-2 from Hugging Face Transformers) are **lazy-loaded** on the first analysis request.
2.  **Perplexity Calculation**: The system uses a pre-trained `gpt2` model to calculate the perplexity of the text. A low perplexity score suggests the text is predictable and likely AI-generated. A fallback unigram model is used if the transformer model fails.
3.  **Burstiness Score**: It analyzes the variation in sentence lengths and word frequency. Human writing tends to have higher "burstiness" (a mix of long and short sentences), while AI text is often more uniform.
4.  **Heuristic Analysis**: Several other signals are computed:
    - **Low Vocabulary Diversity**: AIs often reuse a limited set of words.
    - **Low Sentence Variability**: AI sentences can be monotonous in length.
    - **High Repetition**: Overuse of certain words or phrases.
5.  **Final Verdict**: A weighted score is calculated from all these signals. The application only flags text as "AI-generated" if multiple strong signals are present and the text is of sufficient length (min. 120 tokens), minimizing false positives.

### Plagiarism Checker

The plagiarism detection module identifies similarities between two pieces of text.

1.  **Preprocessing**: Text is cleaned, tokenized, and stopwords are removed.
2.  **Sentence Similarity**: Each sentence from the source text is compared against every sentence in the target text.
3.  **Hybrid Scoring**: Similarity is calculated using a combination of:
    - **TF-IDF Cosine Similarity**: To measure semantic similarity.
    - **N-gram Overlap**: To catch exact phrase matches.
4.  **Overall Score**: The final plagiarism score is an aggregation of the highest similarity scores for each sentence.

---

## 🛠️ Tech Stack

-   **Backend**:
    -   **Framework**: Flask
    -   **ML/NLP**: `Transformers (Hugging Face)`, `PyTorch`, `NLTK`, `Scikit-learn`
    -   **File Handling**: `PyPDF2`, `python-docx`
    -   **Server**: Gunicorn (recommended for production)

-   **Frontend**:
    -   HTML5
    -   CSS3
    -   Vanilla JavaScript (with asynchronous `fetch` for API calls)

-   **Python Version**: 3.8+

---

## 🚀 Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Prerequisites

-   **Python**: Ensure you have Python 3.8 or newer installed. You can check by running `python --version`.
-   **pip**: The Python package installer is required.

### 2. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 3. Install Dependencies

This project uses a `requirements.txt` file to manage its Python dependencies.

```bash
pip install -r requirements.txt
```

The first time you run the application, it will automatically download the necessary `nltk` data models (`punkt`, `stopwords`, etc.).

### 4. Run the Application

The application consists of two parts: the Flask backend and the frontend server. The provided `setup_and_run.bat` script automates this for Windows.

**On Windows:**

Simply double-click the `setup_and_run.bat` file. It will:
1.  Install all dependencies.
2.  Start the Flask backend server.
3.  Start the frontend server.

**Manual Steps (for all Operating Systems):**

You need to run two separate commands in two separate terminals.

**Terminal 1: Start the Backend (Flask)**

```bash
python app.py
```

This will start the backend server, typically on `http://localhost:5000`.

**Terminal 2: Start the Frontend (HTTP Server)**

```bash
cd Front
python -m http.server 8000
```

This will serve the frontend files on `http://localhost:8000`.

### 5. Access the Application

Open your web browser and navigate to:

**[http://localhost:8000](http://localhost:8000)**

---

## 🔌 API Endpoints

The Flask backend provides several API endpoints for text analysis.

-   `POST /check`: The main endpoint for full analysis. Takes a JSON with `{"text": "..."}` and returns perplexity, burstiness, AI likelihood, and other metrics.
-   `POST /upload`: Handles file uploads (`.pdf`, `.docx`, `.txt`). Extracts text and returns it in a JSON response.
-   `POST /analysis`: A simplified endpoint that returns just perplexity and burstiness.
-   `POST /streamlit-analysis`: Returns data formatted for visualization, including word frequency counts.

---

## 📂 Project Structure

```
.
├── Front/                # Contains all frontend files
│   ├── index.html        # Main HTML file
│   ├── styles.css        # CSS for styling
│   └── script.js         # JavaScript for interactivity and API calls
├── __pycache__/          # Python cache
├── app.py                # The core Flask backend application
├── requirements.txt      # List of Python dependencies
├── setup_and_run.bat     # Automation script for Windows
└── README.md             # This file
```
