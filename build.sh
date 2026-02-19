#!/usr/bin/env bash
set -o errexit

# Install all dependencies (torch CPU-only is handled via --extra-index-url in requirements.txt)
pip install -r requirements.txt

# Download required NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('stopwords')"

# Prefetch GPT-2 model and tokenizer into a local cache during build so
# the runtime does not need to download the model on cold start.
# Uses HF_TOKEN environment variable set in Render (export as HF_TOKEN or HUGGINGFACE_HUB_TOKEN).
python - <<'PY'
import os
from pathlib import Path
try:
	# Use a local cache folder inside the project so build artifacts are available at runtime
	cache_dir = Path(os.getcwd()) / '.hf_cache'
	cache_dir.mkdir(parents=True, exist_ok=True)

	# Expose to transformers/huggingface_hub
	os.environ['HF_HOME'] = str(cache_dir)
	# Support both names for token env var
	if 'HF_TOKEN' in os.environ and not os.environ.get('HUGGINGFACE_HUB_TOKEN'):
		os.environ['HUGGINGFACE_HUB_TOKEN'] = os.environ['HF_TOKEN']

	from transformers import AutoTokenizer, AutoModelForCausalLM
	model_name = os.environ.get('MODEL_NAME', os.environ.get('MODEL_NAME', 'gpt2'))
	print('Prefetching model', model_name, 'into', cache_dir)
	# low_cpu_mem_usage reduces peak memory during download/weight sharding
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=str(cache_dir))
	model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, cache_dir=str(cache_dir))
	print('Prefetch complete for', model_name)
except Exception as e:
	print('Prefetch failed (non-fatal):', e)
PY
