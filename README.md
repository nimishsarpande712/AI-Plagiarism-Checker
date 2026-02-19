# AI Plagiarism Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready web application that detects AI-generated text and checks plagiarism using GPT-2 perplexity analysis, burstiness scoring, and TF-IDF similarity. Built for educators, writers, and content creators.

---

## ✨ Features

### Core Analysis
- **AI Text Detection** — Multi-signal engine combining GPT-2 perplexity, burstiness, vocabulary diversity, sentence variability, repetition patterns, and passive voice ratio
- **Plagiarism Checking** — TF-IDF cosine similarity + n-gram overlap between two texts with sentence-level match highlighting
- **File Upload** — Analyze `.pdf`, `.docx`, and `.txt` files directly
- **Sentence-Level Breakdown** — Per-sentence AI probability with perplexity scores

### Advanced Features
- **Cross-Document Analysis** — Compare multiple documents for mutual similarity
- **Batch Analysis** — Upload and analyze multiple files at once
- **Document Compare** — Side-by-side diff view between two texts
- **Feedback System** — Users can flag incorrect results; model adapts via persistent learning
- **Adaptive Learning** — Thresholds auto-adjust based on accumulated feedback stored in Supabase
- **Shareable Reports** — Generate unique report links with view tracking

### Analytics & Visualization
- **Interactive Charts** — Plotly.js graphs for perplexity distribution, word frequency, and style metrics
- **Style Consistency** — Measures writing style uniformity throughout the text
- **Complexity Score** — Estimates text complexity from sentence length and word rarity
- **Trust Score** — Wilson-score confidence interval based on user feedback accuracy

---

## ⚙️ How It Works

### AI Detection Engine

1. **Model Loading** — GPT-2 and NLTK models are lazy-loaded on first request to keep startup fast
2. **Perplexity** — Measures how "surprised" GPT-2 is by the text. Low perplexity = predictable = likely AI-generated. Falls back to a unigram model if transformers fail
3. **Burstiness** — Analyzes sentence-length variation. Humans write with mixed long/short sentences; AI tends to be uniform
4. **Heuristic Signals** — Vocabulary diversity (TTR), sentence variability, word repetition, punctuation patterns, passive voice ratio, n-gram repetition
5. **Weighted Verdict** — All signals are combined with contribution weights. Text must be ≥120 tokens and multiple strong signals must agree before flagging as AI-generated

### Plagiarism Checker

1. **Preprocessing** — Text is cleaned, tokenized, and stopwords are removed
2. **Hybrid Scoring** — Each sentence is scored using TF-IDF cosine similarity (60%) + trigram Jaccard overlap (40%)
3. **Aggregation** — Overall score is the mean of per-sentence max similarities

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Flask, Gunicorn |
| **ML/NLP** | PyTorch, Transformers (GPT-2), NLTK, Scikit-learn |
| **Database** | Supabase (PostgreSQL) |
| **Frontend** | Vanilla JS, Plotly.js, CSS3 |
| **File Parsing** | PyPDF2, python-docx |
| **Hosting** | Render (backend), Vercel (frontend) |

---

## 🚀 Setup

### Prerequisites

- Python 3.8+
- A [Supabase](https://supabase.com) project (free tier works)

### Local Development

```bash
# Clone
git clone https://github.com/nimishsarpande712/AI-Plagiarism-Checker.git
cd AI-Plagiarism-Checker

# Virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Supabase URL and key

# Start backend (http://localhost:5000)
python app.py

# In a new terminal — start frontend (http://localhost:8000)
cd Front
python -m http.server 8000
```

Open **http://localhost:8000** in your browser.

---

## 🌐 Deployment

### Backend → Render

1. Push repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Web Service** → connect your repo
3. Render auto-detects `render.yaml` — accept defaults
4. Set environment variables in the dashboard:

   | Variable | Value |
   |---|---|
   | `SUPABASE_URL` | Your Supabase project URL |
   | `SUPABASE_KEY` | Your Supabase anon key |
   | `ALLOWED_ORIGINS` | Your Vercel frontend URL (e.g. `https://your-app.vercel.app`) |

5. Deploy — copy your Render URL (e.g. `https://ai-plagiarism-checker-xxxx.onrender.com`)

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) → **Add New Project** → import your repo
2. Set **Root Directory** to `Front`
3. Set **Framework Preset** to `Other`
4. Leave build command blank, set output directory to `.`
5. Deploy — copy your Vercel URL

### Connect Them

1. In `Front/index.html`, update the API base URL:
   ```html
   <script>window.__API_BASE__ = "https://ai-plagiarism-checker-xxxx.onrender.com";</script>
   ```
2. In Render dashboard, set `ALLOWED_ORIGINS` to your Vercel URL
3. Commit & push — Vercel auto-redeploys

> **Note:** Render free tier sleeps after 15 min of inactivity. First request after sleep takes ~30-60s. Paid tier ($7/mo) keeps it always-on.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/check` | Full AI detection analysis — returns perplexity, burstiness, AI probability, signals, and verdict |
| `POST` | `/upload` | Extract text from uploaded `.pdf`, `.docx`, or `.txt` file |
| `POST` | `/streamlit-analysis` | Returns perplexity, burstiness, and word frequency data for visualization |
| `POST` | `/sentence-analysis` | Per-sentence AI probability breakdown |
| `POST` | `/cross-document` | Compare multiple documents for mutual similarity |
| `POST` | `/batch-analyze` | Analyze multiple files in one request |
| `POST` | `/compare-texts` | Side-by-side diff between two texts |
| `POST` | `/feedback` | Submit user feedback on analysis accuracy |
| `GET` | `/feedback-stats` | Retrieve feedback accuracy and trust score |
| `POST` | `/reports` | Generate a shareable report link |
| `GET` | `/reports/<token>` | Retrieve a shared report by token |
| `GET` | `/health` | Health check — returns model status |

---

## 📂 Project Structure

```
AI-Plagiarism-Checker/
├── app.py              # Flask backend — all API endpoints and ML logic
├── database.py         # Supabase integration — analyses, reports, feedback, learning
├── requirements.txt    # Python dependencies
├── Procfile            # Gunicorn start command for Render
├── render.yaml         # Render deployment blueprint
├── .env.example        # Environment variable template
├── LICENSE             # MIT License
├── README.md           # This file
└── Front/              # Frontend (deployed to Vercel)
    ├── index.html      # Main HTML
    ├── script.js       # All frontend logic — auth, analysis, compare, batch, charts
    ├── styles.css      # Neo-brutalist UI styling
    ├── vercel.json     # Vercel config — rewrites & security headers
    └── assets/         # Static assets (favicon, images)
```

---

## 📄 License

[MIT](LICENSE)
