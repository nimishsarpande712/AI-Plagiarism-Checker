Hugging Face Space scaffold

This repo contains a Dockerfile and `entrypoint.sh` to run the Flask backend as a Hugging Face Space.

Quick steps to create the Space:

1. Create a new Space on Hugging Face: choose "Custom" or "Docker" and connect to this repository.
2. In the Space settings, add a secret named `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) containing your Hugging Face token. This speeds model downloads.
3. Optionally set `MODEL_NAME` to `gpt2` (default) or another model.
4. Click "Deploy". The container will run `entrypoint.sh`, prefetch the model if `HF_TOKEN` is present, then start the Flask app.

Notes:
- If you set `HF_TOKEN`, the prefetch will run at container start and cache the model in `.hf_cache` inside the container image runtime.
- If prefetch is skipped, the app will download the model at runtime (slower start).
