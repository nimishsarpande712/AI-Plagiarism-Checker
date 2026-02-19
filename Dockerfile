FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy project
COPY . /app

# Install system deps needed for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install python requirements
RUN pip install -r requirements.txt

# Ensure entrypoint is executable
RUN chmod +x /app/entrypoint.sh || true

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Use a local HF cache inside the container
ENV HF_HOME=/app/.hf_cache

# Entrypoint handles optional model prefetch (uses HF_TOKEN/HUGGINGFACE_HUB_TOKEN)
CMD ["/app/entrypoint.sh"]
