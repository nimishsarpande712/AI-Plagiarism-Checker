#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt

python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
"
