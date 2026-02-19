#!/usr/bin/env bash
set -euo pipefail

# If HF_TOKEN is provided, expose it as HUGGINGFACE_HUB_TOKEN for huggingface libs
if [ -n "${HF_TOKEN-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "Starting entrypoint: prefetch (if token present) then run gunicorn"

python - <<'PY'
import os
from pathlib import Path
try:
    cache_dir = Path(os.environ.get('HF_HOME', '/app/.hf_cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_name = os.environ.get('MODEL_NAME', 'gpt2')
    print('Prefetching model', model_name, 'into', cache_dir)
    # Ensure transformers sees the cache dir
    os.environ['HF_HOME'] = str(cache_dir)
    # Only attempt prefetch if a token is available to speed downloads
    if os.environ.get('HUGGINGFACE_HUB_TOKEN'):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=str(cache_dir))
        AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, cache_dir=str(cache_dir))
        print('Prefetch complete for', model_name)
    else:
        print('No HUGGINGFACE_HUB_TOKEN found; skipping prefetch (will download at runtime)')
except Exception as e:
    print('Prefetch step failed (continuing):', e)
PY

# Start the app using gunicorn; use the PORT env var provided by Spaces
exec gunicorn app:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 1 --timeout 300
